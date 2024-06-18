import numpy as np
import torch
import torch.nn.functional as F


class RunningCalibration():
    def __init__(self, num_classes, num_bins):
        self.num_bins = num_bins
        self.num_classes = num_classes
        self.tot_intersect_mc = np.zeros((num_classes, num_bins))
        self.tot_cert_mc = np.zeros((num_classes, num_bins))
        self.tot_weight_mc = np.zeros((num_classes, num_bins))
        self.tot_intersect = np.zeros(( num_bins))
        self.tot_cert = np.zeros((num_bins))
        self.tot_weight = np.zeros((num_bins))
    def update_multiclass_ece(self, result, label, cert):
        for i in range(self.num_classes):
            msk = label == i
            class_intersect, class_cert, class_weight = self.multiclass_ece(result[msk], label[msk], cert[msk])
            self.tot_intersect_mc[i] += class_intersect
            self.tot_cert_mc[i] += class_cert
            self.tot_weight_mc[i] += class_weight

    

    def multiclass_ece(self, result, label, cert): 
        cert[cert>=1] = 1.0
        bins = np.linspace(0, 1, self.num_bins + 1)
        class_intersect = np.zeros(self.num_bins)
        class_cert = np.zeros(self.num_bins)
        class_weight = np.zeros(self.num_bins)
        for i in range(len(bins)-1):
            inx_l = cert>bins[i]
            inx_h = cert<=bins[i+1]
            inx = inx_l * inx_h
            t_label = label[inx]
            t_result = result[inx]
            t_cert = cert[inx]
            class_intersect[i]+= (t_label == t_result).sum()
            class_cert[i]+= t_cert.sum()
            class_weight[i]+=t_label.size
        return class_intersect, class_cert, class_weight

    def update_ece_mce(self, result, label, cert):            
        cert[cert>=1] = 1.0
        bins = np.linspace(0, 1, self.num_bins + 1)
        for i in range(len(bins)-1):
            inx_l = cert>bins[i]
            inx_h = cert<=bins[i+1]
            inx = inx_l * inx_h
            t_label = label[inx]
            t_result = result[inx]
            t_cert = cert[inx]
            self.tot_intersect[i]+= (t_label == t_result).sum()
            self.tot_cert[i]+= t_cert.sum()
            self.tot_weight[i]+=t_label.size
    def get_multiclass_ece(self):
        tot_ece = 0
        for i in range(self.num_classes):
            tot_ece += np.nan_to_num(self.multicass_ece_calc(self.tot_weight_mc[i], self.tot_intersect_mc[i], self.tot_cert_mc[i]))
        return np.round(tot_ece/self.num_classes, 2)
    def multicass_ece_calc(self, tot_weight, tot_intersect, tot_cert):
        prop_bin = tot_weight/tot_weight.sum()
        acc_bin = np.nan_to_num(tot_intersect/tot_weight)
        conf_bin = np.nan_to_num(tot_cert/tot_weight)
        return np.sum(np.absolute(acc_bin - conf_bin) * prop_bin)*100
    def get_ece(self):
        prop_bin = self.tot_weight/self.tot_weight.sum()
        acc_bin = np.nan_to_num(self.tot_intersect/self.tot_weight)
        conf_bin = np.nan_to_num(self.tot_cert/self.tot_weight)
        return np.round(np.sum(np.absolute(acc_bin - conf_bin) * prop_bin)*100, 2)
    def get_mce(self):
        acc_bin = np.nan_to_num(self.tot_intersect/self.tot_weight)
        conf_bin = np.nan_to_num(self.tot_cert/self.tot_weight)
        return np.round(np.max(np.absolute(acc_bin - conf_bin))*100, 2)





class RunningMetrics():
    def __init__(self, num_class):
        self.num_class = num_class
        self.class_int = np.zeros(num_class)
        self.class_uni = np.zeros(num_class)
        self.class_tot = np.zeros(num_class)
        self.overall_int = 0
        self.overall_tot = 0

    def update_acc_miou(self, result, label):
        self.overall_int += (label==result).sum()
        self.overall_tot += label.shape[0]
        for i in range(self.num_class):
            label_i = label==i
            label_p_i = result==i
            self.class_int[i] += (label_i*label_p_i).sum()
            self.class_uni[i] += (label_i+label_p_i).sum()
            self.class_tot[i] += label_i.sum()
    
    def get_aacc(self):
        return np.round((self.overall_int/self.overall_tot)*100, 2)
    def get_macc(self):
        cls_msk = self.class_int>0
        acc_class = (self.class_int[cls_msk]/self.class_tot[cls_msk])*100
        return np.round(np.mean(acc_class), 2)
    def get_miou(self):
        cls_msk = self.class_int>0
        iou_class = (self.class_int[cls_msk]/self.class_uni[cls_msk])*100
        return np.round(np.mean(iou_class), 2)

class RunningPredictions():
    def __init__(self, num_class, T):
        self.num_class = num_class
        self.T = T

    
    def pred_childred(self, logits):
        seg_confs = F.softmax(logits[:,0:19]/self.T, dim=1)
        certs, preds = self.final_prediction(seg_confs)
        return certs, preds
    def pred_parents(self, logits):
        parent_logits = logits[:,19:26]
        seg_confs = F.softmax(parent_logits/self.T, dim=1)
        certs, preds = self.final_prediction(seg_confs)
        return certs, preds
    def pred_parents_together(self, logits):
        seg_confs = F.softmax(logits/self.T, dim=1)
        p1 = seg_confs[:,0:2].sum(1).unsqueeze(1) + seg_confs[:,-7].unsqueeze(1)
        p2 = seg_confs[:,2:5].sum(1).unsqueeze(1) + seg_confs[:,-6].unsqueeze(1)
        p3 = seg_confs[:,5:8].sum(1).unsqueeze(1) + seg_confs[:,-5].unsqueeze(1)
        p4 = seg_confs[:,8:10].sum(1).unsqueeze(1) + seg_confs[:,-4].unsqueeze(1)
        p5 = seg_confs[:,10:11].sum(1).unsqueeze(1) + seg_confs[:,-3].unsqueeze(1)
        p6 = seg_confs[:,11:13].sum(1).unsqueeze(1)  + seg_confs[:,-2].unsqueeze(1)
        p7 = seg_confs[:,13:19].sum(1).unsqueeze(1) + seg_confs[:,-1].unsqueeze(1)
        seg_confs = torch.cat([p1, p2, p3, p4, p5, p6, p7], dim=1)
        certs, preds = self.final_prediction(seg_confs)
        return certs, preds

    def infer_parents(self, logits):
        seg_confs = F.softmax(logits[:,0:19]/self.T, dim=1)
        p1 = seg_confs[:,0:2].sum(1).unsqueeze(1)
        p2 = seg_confs[:,2:5].sum(1).unsqueeze(1)
        p3 = seg_confs[:,5:8].sum(1).unsqueeze(1)
        p4 = seg_confs[:,8:10].sum(1).unsqueeze(1)
        p5 = seg_confs[:,10:11].sum(1).unsqueeze(1)
        p6 = seg_confs[:,11:13].sum(1).unsqueeze(1)
        p7 = seg_confs[:,13:19].sum(1).unsqueeze(1)
        seg_confs = torch.cat([p1, p2, p3, p4, p5, p6, p7], dim=1)
        certs, preds = self.final_prediction(seg_confs)
        return certs, preds

    def pred_together(self, logits):
        seg_confs = F.softmax(logits/self.T, dim=1)
        seg_confs = self.add_parents(seg_confs)
        certs, preds = self.final_prediction(seg_confs)
        return certs, preds

    def final_prediction(self, seg_confs):
        
        seg_confs = seg_confs.mean(0).unsqueeze(0)
        certs, preds = torch.max(seg_confs, dim=1)
        return preds.squeeze().cpu().numpy(), certs.squeeze().cpu().numpy()

    def add_parents(self, confs):
        confs[:,0:2]+=confs[:,-7].unsqueeze(1)
        confs[:,2:5]+=confs[:,-6].unsqueeze(1)
        confs[:,5:8]+=confs[:,-5].unsqueeze(1)
        confs[:,8:10]+=confs[:,-4].unsqueeze(1)
        confs[:,10:11]+=confs[:,-3].unsqueeze(1)
        confs[:,11:13]+=confs[:,-2].unsqueeze(1)
        confs[:,13:19]+=confs[:,-1].unsqueeze(1)
        return confs



    

def transform_l2_label(label):
    l2_label = np.zeros(label.shape)
    l2_label[(label == 0) + (label == 1)] = 0
    l2_label[(label == 2) + (label == 3) + (label == 4)] = 1
    l2_label[(label == 5) + (label == 6) + (label == 7)] = 2
    l2_label[(label == 8) + (label == 9)] = 3
    l2_label[(label == 10)] = 4
    l2_label[(label == 11) + (label == 12)] = 5
    l2_label[(label == 13) + (label == 14) + (label == 15) + (label == 16) + (label == 17) + (label == 18)] = 6
    return l2_label
