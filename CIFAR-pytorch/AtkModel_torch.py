# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 01:12:20 2023

@author: omars
"""
import torch.nn as nn
import torch
import numpy as np
import pickle
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
import torch.nn.functional as F

class WhiteBoxAttackModel(nn.Module):
    def __init__(self, class_num, total, intermediate_layer_input,method,device,l1=128,l2=64):
        super(WhiteBoxAttackModel, self).__init__()

        self.intermediate_layer_input = intermediate_layer_input
        self.method=method
        self.device=device

        self.Intermediate_Output_Component_result = nn.Sequential(
                nn.Dropout(p=0.2),
                # nn.Linear(class_num, 128),
                nn.Linear(self.intermediate_layer_input, l1),
                nn.ReLU(),
                nn.Linear(l1, l2)
            )

        self.Output_Component = nn.Sequential(
			nn.Dropout(p=0.2),
			nn.Linear(class_num, 128),
			nn.ReLU(),
			nn.Linear(128, 64),
		)

        self.Loss_Component = nn.Sequential(
			nn.Dropout(p=0.2),
			nn.Linear(1, 128),
			nn.ReLU(),
			nn.Linear(128, 64),
		)

        self.Label_Component = nn.Sequential(
			nn.Dropout(p=0.2),
			nn.Linear(class_num, 128),
			nn.ReLU(),
			nn.Linear(128, 64),
		)

        self.Encoder_Component = nn.Sequential(
            nn.Dropout(p=0.2),
			nn.Linear(320, 256),
			nn.ReLU(),
            nn.Dropout(p=0.2),
			nn.Linear(256, l1),
			nn.ReLU(),
			nn.Dropout(p=0.2),
			nn.Linear(l1, l2),
			nn.ReLU(),
			nn.Linear(l2, 2),
		)


    def forward(self, intermediate_output, output, loss, gradient, label):

        if self.method != "NoIntermediate":
            try:
                intermediate_output = intermediate_output.to(self.device)
                self.intermediate_layer_input = intermediate_output.shape[0]

                self.Intermediate_Output_Component_result = nn.Sequential(
                    nn.Dropout(p=0.2),
                    nn.Linear(self.intermediate_layer_input, 128),
                    nn.ReLU(),
                    nn.Linear(128, 64),
                )
                self.Intermediate_Output_Component_result = self.Intermediate_Output_Component_result.to(self.device)

                Intermediate_Output_Component_result = self.Intermediate_Output_Component_result(intermediate_output)
            except:
                intermediate_output = np.transpose(intermediate_output.cpu().detach().numpy())
                intermediate_output = torch.squeeze(torch.Tensor(intermediate_output), 1)
                intermediate_output = intermediate_output.to(self.device)

                self.intermediate_layer_input = intermediate_output.shape[1]

                self.Intermediate_Output_Component_result = nn.Sequential(
                    nn.Dropout(p=0.2),
                    nn.Linear(self.intermediate_layer_input, 128),
                    nn.ReLU(),
                    nn.Linear(128, 64),
                )
                self.Intermediate_Output_Component_result = self.Intermediate_Output_Component_result.to(self.device)

                Intermediate_Output_Component_result = self.Intermediate_Output_Component_result(intermediate_output)
        else:
            Output_Component_result = self.Output_Component(output)

        Loss_Component_result = self.Loss_Component(loss)
        Gradient_Component_result = self.Gradient_Component(gradient)
        Label_Component_result = self.Label_Component(label)

        Output_Component_result = self.Output_Component(output)
        final_inputs = torch.cat((Intermediate_Output_Component_result, Output_Component_result, Loss_Component_result, Gradient_Component_result, Label_Component_result), 1)
        #final_inputs = torch.cat((Output_Component_result, Loss_Component_result, Gradient_Component_result, Label_Component_result), 1)

        final_result = self.Encoder_Component(final_inputs)
        return final_result

    def train(self, epoch, result_path):
        self.attack_model.train()
        batch_idx = 1
        train_loss = 0
        correct = 0
        total = 0

        final_train_gndtrth = []
        final_train_predict = []
        final_train_probabe = []

        final_result = []

        with open(self.ATTACK_SETS + "train.p", "rb") as f:
            while(True):
                try:
                    intermediate_output, output, loss, gradient, label, members = pickle.load(f)
                    output, loss, gradient, label, members = output.to(self.device), loss.to(self.device), gradient.to(self.device), label.to(self.device), members.to(self.device)
                    # print("self.attack_model", self.attack_model.device)

                    results = self.attack_model(intermediate_output, output, loss, gradient, label)
                    # results = F.softmax(results, dim=1)
                    losses = self.attack_criterion(results, members)

                    losses.backward()
                    self.optimizer.step()

                    train_loss += losses.item()
                    _, predicted = results.max(1)
                    total += members.size(0)
                    correct += predicted.eq(members).sum().item()

                    if epoch:
                        final_train_gndtrth.append(members)
                        final_train_predict.append(predicted)
                        final_train_probabe.append(results[:, 1])

                    batch_idx += 1
                except EOFError:
                    break

        if epoch:
            final_train_gndtrth = torch.cat(final_train_gndtrth, dim=0).cpu().detach().numpy()
            final_train_predict = torch.cat(final_train_predict, dim=0).cpu().detach().numpy()
            final_train_probabe = torch.cat(final_train_probabe, dim=0).cpu().detach().numpy()

            conf_matrix =  confusion_matrix(final_train_gndtrth, final_train_predict)

            train_f1_score = f1_score(final_train_gndtrth, final_train_predict)
            train_roc_auc_score = roc_auc_score(final_train_gndtrth, final_train_probabe)

            final_result.append(train_f1_score)
            final_result.append(train_roc_auc_score)

            with open(result_path, "wb") as f:
                pickle.dump((final_train_gndtrth, final_train_predict, final_train_probabe), f)

            print("Saved Attack Train Ground Truth and Predict Sets")
            print("Train F1: %f\nAUC: %f" % (train_f1_score, train_roc_auc_score))

            print("\n\n\n\n")
            tn, fp, fn, tp = conf_matrix.ravel()
            print( 'Train Acc: %.3f%% (%d/%d)' % (100.*correct/(1.0*total), correct, total))
            print("Recall:", str(round((tp/(tp+fn)) * 100, 2)))
            print("Negative Recall:", str(round((tn/(tn+fp)) * 100, 2)))
            print("Train F1: %f\nAUC: %f" % (train_f1_score, train_roc_auc_score))
            print("TP: ", tp, "FP", fp, "TN", tn, "FN", fn)
            print(round((100.*correct)/(1.0*total),2),"\t",round((tp/(tp+fn))*100, 2),"\t",round((tn/(tn+fp))*100, 2),"\t", round(train_f1_score*100,2),"\t", round(train_roc_auc_score*100,2))
            self.results['train_accuracy'].append(round((100.*correct)/(1.0*total),2))
            self.results['train_recall'].append(round((tp/(tp+fn))*100, 2))
            self.results['train_negative_recall'].append(round((tn/(tn+fp))*100, 2))
            self.results['train_f1'].append(round(train_f1_score*100,2))
            self.results['train_auc'].append(round(train_roc_auc_score*100,2))


        try:
            final_result.append(1.*correct/total)
        except:
            final_result.append(1.*correct/1)


        print( 'Train Acc: %.3f%% (%d/%d) | Loss: %.3f' % (100.*correct/total, correct, total, 1.*train_loss/batch_idx))


        return final_result




    def test(self, epoch, result_path):
        self.attack_model.eval()
        batch_idx = 1
        correct = 0
        total = 0

        final_test_gndtrth = []
        final_test_predict = []
        final_test_probabe = []

        final_result = []

        with torch.no_grad():
            with open(self.ATTACK_SETS + "test.p", "rb") as f:
                while(True):
                    try:
                        intermediate_output, output, loss, gradient, label, members = pickle.load(f)
                        output, loss, gradient, label, members = output.to(self.device), loss.to(self.device), gradient.to(self.device), label.to(self.device), members.to(self.device)

                        results = self.attack_model(intermediate_output, output, loss, gradient, label)

                        _, predicted = results.max(1)
                        total += members.size(0)
                        correct += predicted.eq(members).sum().item()

                        results = F.softmax(results, dim=1)

                        if epoch:
                            final_test_gndtrth.append(members)
                            final_test_predict.append(predicted)
                            final_test_probabe.append(results[:, 1])

                        batch_idx += 1
                    except EOFError:
                        break

        if epoch:
            final_test_gndtrth = torch.cat(final_test_gndtrth, dim=0).cpu().numpy()
            final_test_predict = torch.cat(final_test_predict, dim=0).cpu().numpy()
            final_test_probabe = torch.cat(final_test_probabe, dim=0).cpu().numpy()

            conf_matrix = confusion_matrix(final_test_gndtrth, final_test_predict)

            test_f1_score = f1_score(final_test_gndtrth, final_test_predict)
            test_roc_auc_score = roc_auc_score(final_test_gndtrth, final_test_probabe)

            final_result.append(test_f1_score)
            final_result.append(test_roc_auc_score)


            with open(result_path, "wb") as f:
                pickle.dump((final_test_gndtrth, final_test_predict, final_test_probabe), f)

            print("Saved Attack Test Ground Truth and Predict Sets")
            print("Test F1: %f\nAUC: %f" % (test_f1_score, test_roc_auc_score))

            print("\n\n\n\n")
            tn, fp, fn, tp = conf_matrix.ravel()
            print( 'Test Acc: %.2f%% (%d/%d)' % (100.*correct/(1.0*total), correct, total))
            print("Recall:", str(round((tp/(tp+fn)) * 100, 2)))
            print("Negative Recall:", str(round((tn/(tn+fp)) * 100, 2)))
            print("Test F1: %.2f\nAUC: %.2f" % (test_f1_score*100, test_roc_auc_score*100))
            print("TP: ", tp, "FP", fp, "TN", tn, "FN", fn)
            print(round((100.*correct)/(1.0*total),2),"\t",round((tp/(tp+fn))*100, 2),"\t",round((tn/(tn+fp))*100, 2),"\t", round(test_f1_score*100,2),"\t", round(test_roc_auc_score*100,2))

            self.results['test_accuracy'].append(round((100.*correct)/(1.0*total),2))
            self.results['test_recall'].append(round((tp/(tp+fn))*100, 2))
            self.results['test_negative_recall'].append(round((tn/(tn+fp))*100, 2))
            self.results['test_f1'].append(round(test_f1_score*100,2))
            self.results['test_auc'].append(round(test_roc_auc_score*100,2))
            self.save_results()

        final_result.append(1.*correct/total)

        print( 'Test Acc: %.3f%% (%d/%d)' % (100.*correct/(1.0*total), correct, total))


        return final_result