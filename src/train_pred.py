import time
import torch
import copy
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import optuna
from collections import Counter 
import logging
from sklearn.metrics import roc_auc_score, confusion_matrix
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans

device = torch.device("cuda:0" if torch.cuda.is_available()
                                   else "cpu")

def train_model(model, criterion, optimizer, dataloaders, dataset_sizes,
                lr_scheduler, save_path, name, num_epochs=25, verbose=True):
    LOG = save_path + "/execution-" + name + ".log" 
    logging.basicConfig(filename=LOG, filemode="w", level=logging.DEBUG)  

    # console handler  
    console = logging.StreamHandler()  
    console.setLevel(logging.ERROR)  
    logging.getLogger("").addHandler(console)

    logger = logging.getLogger(__name__)
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_epoch = 0
    best_auc = 0

    acc_array = {'train': [], 'val': []}
    loss_array = {'train': [], 'val': []}
    auc_array = {'train': [], 'val': []}
    best_val_preds = []
    best_train_preds = []
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        logger.debug('Epoch {}/{}'.format(epoch, num_epochs - 1))
        logger.debug('-' * 10)
        val_preds = []
        train_preds = []
        train_case_ids = []
        train_labels = []
        val_labels = []
        train_labels_auc = []
        train_preds_auc = []
        val_labels_auc = []
        val_preds_auc = []
        sizes = {'train': 0, 'val': 0}
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels, case_ids in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                inputs.requires_grad = True
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    _, mlabels = torch.max(labels, 1)

                    loss = criterion(outputs, mlabels)
                    if phase == 'val':
                        val_preds += list(preds.cpu().numpy())
                        val_labels += list(mlabels.cpu().numpy())
                        val_labels_auc += [labels.cpu().numpy()]
                        val_preds_auc += [preds.cpu().numpy()]

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        train_preds += list(preds.cpu().numpy())
                        train_case_ids += [case_ids]
                        train_labels += list(mlabels.cpu().numpy())
                        train_labels_auc += [labels.cpu().numpy()]
                        train_preds_auc += [preds.cpu().numpy()]
                        loss.backward()
                        optimizer.step()
                
                # statistics

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == mlabels)
                sizes[phase] += inputs.size(0)
                
            epoch_loss = running_loss / sizes[phase]
            epoch_acc = running_corrects.item() / sizes[phase]

            loss_array[phase].append(epoch_loss)
            acc_array[phase].append(epoch_acc)
            
            if verbose:
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))
                logger.debug('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_epoch = epoch
                best_acc = epoch_acc
                #best_auc = epoch_auc
                best_model_wts = copy.deepcopy(model.state_dict())
                best_val_preds = val_preds[:]
                best_train_preds = train_preds[:]
        print()

    time_elapsed = time.time() - since
    if verbose:
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        logger.debug('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        logger.debug('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    if best_val_preds == []:
        best_val_preds = val_preds[:]
    results = {
        'model': model,
        'best_acc': best_acc,
        #'best_auc': best_auc,
        'best_epoch': best_epoch,
        'acc_array': acc_array,
        'loss_array': loss_array,
        'auc_array': auc_array,
        'val_preds': best_val_preds,
        'val_labels': val_labels,
        'train_preds': best_train_preds,
        'train_case_ids': train_case_ids,
        'train_labels': train_labels
    }

    return results

def predict_patches(model, dataloader, dataset_size):
    from sklearn.metrics import accuracy_score
    model.eval()
    val_preds = []
    val_labels = []

    for inputs, labels, caseids in dataloader:
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            _, mlabels = torch.max(labels, 1)
            
            val_preds += list(preds.cpu().numpy())
            val_labels += list(mlabels.cpu().numpy())
    
    acc = accuracy_score(val_labels, val_preds)

    print('Accuracy {:.4f}'.format(acc))

    return acc, val_preds, val_labels
    
def predict_WSI(model, dataloader, dataset_size, verbose=True):
        """Predict an image by using patches."""
        activation = {}
        def _get_features(name):
            def hook(model, input, output):
                activation[name] = input[0].detach()
            return hook

        model.eval()
        since = time.time()
        corrects = 0
        # results variables
        preds = []
        patch_preds = []
        probs = {'LUAD': [], 'HLT': [], 'LUSC': []}
        model.fc.register_forward_hook(_get_features('fc'))
        features = []
        case_ids = []
        # a prediction is going to be made for each patch,
        # and the final prediction will be the most predicted class
        sizes = 0
        i = 0
        patch_labels = []
        global_labels = []
        for inputs, labels, cids in tqdm(dataloader):
            _, mlabel = torch.max(labels, 1)
            predictions = []
            local_features = []
            for patch, cid in zip(inputs, cids):
                patch = patch.to(device).float() 
                output = model(patch)
                _, pred = torch.max(output, 1)
                predictions.append(pred.item())
                local_features.append(activation['fc'].cpu().numpy())
                case_ids.append(cid)
                patch_labels.append(mlabel.item())

            # obtaining final prediction
            final_preds = Counter(predictions)           
            if mlabel.item() == final_preds.most_common(1)[0][0]:
                corrects += 1
            
            #saving preds, labels, features
            preds.append(final_preds.most_common(1)[0][0])
            patch_preds.append(predictions)
            global_labels.append(mlabel.item())
            features.append(local_features)
            
            # saving probabilities
            if 0 in final_preds.keys():
                probs['LUAD'].append(final_preds[0]/len(inputs))
            else:
                probs['LUAD'].append(0)
            if 1 in final_preds.keys():
                probs['HLT'].append(final_preds[1]/len(inputs))
            else:
                probs['HLT'].append(0)
            if 2 in final_preds.keys():
                probs['LUSC'].append(final_preds[2]/len(inputs))
            else:
                probs['LUSC'].append(0)
            
            del inputs, labels


        acc = corrects / dataset_size

        time_elapsed = time.time() - since
        if verbose:
            print('Test complete in {:.0f}m {:.0f}s'.format(
                  time_elapsed // 60, time_elapsed % 60))
        
        all_results = {
            'acc': acc,
            'preds': preds,
            'patch_preds': patch_preds,
            'patch_labels': patch_labels,
            'global_labels': global_labels,
            'probs': probs,
            'patch_case_ids': case_ids,
            'features': features
        }

        return all_results

def predict_WSI_gen(model, dataloader, dataset_size, verbose=True):
        """Predict an image by using patches."""
        since = time.time()
        corrects = 0
        test_preds = []
        model.eval()
        sizes = 0
        i = 0
        test_labels = []
        with torch.set_grad_enabled(False):
            for inputs, labels, case_ids in tqdm(dataloader):
                predictions = []
                for patch in inputs[0]:
                    patch = patch.to(device).float() 
                    output = model.fc(patch)
                    _, pred = output.max(0)
                    predictions.append(pred.item())
                final_pred = Counter(predictions).most_common(1)[0][0]

                if labels.item() == final_pred:
                    corrects += 1
                
                test_preds.append(final_pred)
                test_labels.append(labels.item())
                
                del inputs, labels
        acc = corrects / dataset_size

        time_elapsed = time.time() - since
        if verbose:
            print('Test complete in {:.0f}m {:.0f}s'.format(
                  time_elapsed // 60, time_elapsed % 60)) 
        return acc, test_preds

def obtain_WSI_features(model, dataloader, method='gap'):
        """Obtain features from patches.

        Args:
            batch_X (np.ndarray): array of numpy arrays of data
        """
        activation = {}
        def _get_features(name):
            def hook(model, input, output):
                activation[name] = input[0].detach()
            return hook

        model.eval()
        model.fc.register_forward_hook(_get_features('fc'))

        features_img = []
        features_gen = []

        for inputs, gen, labels in tqdm(dataloader):
            _, mlabel = torch.max(labels, 1)
            local_features = torch.Tensor().to(device)

            for patch in inputs:
                patch = patch.to(device).float() 
                output = model(patch)
                img_features = torch.cat([local_features, activation['fc']], dim=0)

            # global average pooling
            if method == 'gap':
                img_features = img_features.view(1, img_features.shape[1], 1, img_features.shape[0])
                gap = img_features.mean([2, 3])
                features_img.append(gap.cpu().numpy())
                features_gen.append(gen)
            # using kmeans to obtain three centroids and use those features.
            elif method == 'clusters':
                kmeans = KMeans(n_clusters=3, random_state=0).fit(img_features)
                centroids = torch.from_numpy(kmeans.cluster_centers_.flatten()).to(device)
                combined_features = torch.cat([gen.to(device), centroids], dim=1)
                features = torch.cat([features, combined_features], dim=0)
            
            del inputs, labels, img_features

        return features_img, features_gen

def obtain_patch_gen_features(model, dataloader):
        """Obtain features from patches and gene expression.
        """
        activation = {}
        def _get_features(name):
            def hook(model, input, output):
                activation[name] = input[0].detach()
            return hook

        model.eval()
        model.fc.register_forward_hook(_get_features('fc'))

        features_img = []
        features_gen = []
        labels_all = []
        case_ids =  []

        for inputs, gen, labels, c_id in tqdm(dataloader):
            _, mlabel = torch.max(labels, 1)
            local_features = torch.Tensor().to(device)
            case_ids += [c_id]

            for patch in inputs:
                patch = patch.to(device).float() 
                output = model(patch)
                features_img.append(activation['fc'].cpu().numpy())
                features_gen.append(gen)
                labels_all.append(mlabel.cpu().numpy())
           
            del inputs

        return features_img, features_gen, labels_all, case_ids

def get_probs(model, dataloader, verbose=True):
        model.eval()
        test_probs = {'LUAD': [], 'HLT': [], 'LUSC': []}
        all_labels = []
        with torch.set_grad_enabled(False):
            for inputs, labels, _ in tqdm(dataloader):
                predictions = []
                for patch in inputs[0]:
                    patch = patch.to(device).float() 
                    output = model.fc(patch)
                    out_prob = torch.nn.functional.softmax(output, dim=0)
                    _, pred = out_prob.max(0)
                    predictions.append(pred.item())
                
                final_preds = Counter(predictions)
                if 0 in final_preds.keys():
                    test_probs['LUAD'].append(final_preds[0]/inputs.shape[1])
                else:
                    test_probs['LUAD'].append(0)
                if 1 in final_preds.keys():
                    test_probs['HLT'].append(final_preds[1]/inputs.shape[1])
                else:
                    test_probs['HLT'].append(0)
                if 2 in final_preds.keys():
                    test_probs['LUSC'].append(final_preds[2]/inputs.shape[1])
                else:
                    test_probs['LUSC'].append(0)
                
                all_labels.append(labels.item())
                del inputs, labels
        
        return test_probs, all_labels
        