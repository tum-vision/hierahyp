from PIL import Image
import os.path as osp
import numpy as np
import torch
from torch.nn.functional import interpolate as interp
from preperation.json2labelImg import json2labelImg
from tqdm import tqdm
import os

def create_folder(dataset_name, save_dir):
    data_dir = osp.join(save_dir, dataset_name)
    if not osp.exists(data_dir):
        os.makedirs(data_dir)
        os.makedirs(osp.join(data_dir, r"image"))
        os.makedirs(osp.join(data_dir, r"label"))
    return osp.join(data_dir, r"image"), osp.join(data_dir, r"label")



def process_idd(root, map_, dataset_name,  save_dir):
    image_save_dir, label_save_dir = create_folder(dataset_name, save_dir)
    img_path = osp.join(root, r'leftImg8bit\val')
    ply_path = osp.join(root,r'gtFine\val')
    all_images=[]
    all_ply = []
    img_paths_pre = os.listdir(img_path)
    for dirx in img_paths_pre:
        dir_img_path = os.listdir(img_path + r"/" + dirx)
        for dirxy in dir_img_path:
            poly_path = dirxy.split("_", 1)[0]
            all_images.append(img_path + r"/" + dirx+ r"/" + dirxy)
            all_ply.append(ply_path + r"/" + dirx+ r"/" + poly_path + "_gtFine_polygons.json")
    for i, img in enumerate(tqdm(all_images)):
        image = Image.open(all_images[i])
        label = np.asarray(json2labelImg(all_ply[i], None, "level3Id"))
        label_copy = 255 * np.ones(label.shape, dtype=int)
        for k, v in map_.items():
            if k != "null":
                label_copy[label == int(k)] = v
        img_l = Image.fromarray(label_copy)
        image.save(osp.join(image_save_dir, dataset_name + "_" + str(i) + ".png"))
        img_l.save(osp.join(label_save_dir, dataset_name + "_" + str(i) + ".png"))





def process_wilddash(root, map_, dataset_name, save_dir):
    image_save_dir, label_save_dir = create_folder(dataset_name, save_dir)
    with open(os.getcwd() + '/validation.txt', 'r') as file:
    # Read names of validation images
        lines = file.readlines()
    images = np.array([line[:-1] + ".jpg" for line in lines])
    semantic = np.array([line[:-1] + "_labelIds.png" for line in lines])
    image_dir = osp.join(root, r'images')
    label_dir = osp.join(root, r'semantic')
    #label conversion and saving
    for i, img in enumerate(tqdm(images)):
        np_sem = np.asarray(Image.open(osp.join(label_dir, semantic[i])))
        np_img = Image.open(osp.join(image_dir, images[i]))
        label_copy = 255 * np.ones(np_sem.shape, dtype=np.float32)
        for k, v in map_.items():
            label_copy[np_sem == int(k)] = v
        label_copy[label_copy<0] = 255

        im_l = Image.fromarray(np.uint8(label_copy))
        im_l.save(osp.join(label_save_dir, dataset_name + "_" + str(i) + ".png"))
        np_img.save(osp.join(image_save_dir, dataset_name + "_" + str(i) + ".jpg"))

def process_acdc(root, map_, dataset_name,  save_dir):
    conditions = ["fog", "night", "rain", "snow"]
    image_save_dir, label_save_dir = create_folder(dataset_name, save_dir)
    image_dir = osp.join(root, r"rgb_anon")
    label_dir = osp.join(root, r"gt_trainval/gt")


    #collect acdc paths for images and labels. acdc stores images separately for different conditions. 
    all_images = []
    all_labels = []
    all_masks = []
    for c in conditions:
        condition_images = []
        condition_labels = []
        cur_path = os.listdir(image_dir + "/" + c + "/val")
        for path in cur_path:
            condition_images.extend(os.listdir(image_dir + "/" + c + "/val/" + path))
        all_images.extend([image_dir + "/" + c + "/val/" + im.split("_", 1)[0] + "/" + im for im in condition_images ])
        all_labels.extend(([label_dir + "/" + c + "/val/" +  im.split("_", 1)[0] + "/" + im.split("_", 3)[0] + "_" +
                            im.split("_", 3)[1] + "_" + im.split("_", 3)[2] + "_gt_labelIds.png" for im in condition_images ]))
        all_masks.extend(([label_dir + "/" + c + "/val/" +  im.split("_", 1)[0] + "/" + im.split("_", 3)[0] + "_" +
                            im.split("_", 3)[1] + "_" + im.split("_", 3)[2] + "_gt_invIds.png" for im in condition_images ]))
    #label conversion and saving. bring all images and labels under the same directory.
    for i, img in enumerate(tqdm(all_images)):
        image = Image.open(all_images[i])
        label = np.asarray(Image.open(all_labels[i]))
        mask = np.asarray(Image.open(all_masks[i]))
        label_conv = 255 * np.ones(label.shape, dtype=np.int32)
        for k, v in map_.items():
            label_conv[label == int(k)] = v
        label_conv[mask == 1] = 255
        img_m = Image.fromarray(label_conv)
        img_m.save(osp.join(label_save_dir, dataset_name + "_" + str(i) + ".png"))
        image.save(osp.join(image_save_dir, dataset_name + "_" + str(i) + ".png"))

def process_bdd(root, map_, dataset_name,  save_dir):
    image_save_dir, label_save_dir = create_folder(dataset_name, save_dir)
    

    image_dir = osp.join(root, r"images/10k/val")
    label_dir = osp.join(root, r"labels/sem_seg/masks/val")
    imgs = np.array(os.listdir(image_dir))
    #simply save existing labels and images to the folder, with required naming
    for i, img in enumerate(tqdm(imgs)):
        Image.open(os.path.join(image_dir, img)).save(osp.join(image_save_dir, dataset_name + "_" + str(i) + ".jpg"))
        Image.open(os.path.join(label_dir, img.split(".")[0] + ".png")).save(osp.join(label_save_dir, dataset_name + "_" + str(i) + ".png"))


def process_mapillary(root, map_, dataset_name, save_dir):
    image_save_dir, label_save_dir = create_folder(dataset_name, save_dir)
    image_dir = osp.join(root, r"validation/images")
    label_dir = osp.join(root, r"validation/v1.2/labels")
    imgs = np.array(os.listdir(image_dir))

    #label conversion and saving
    for i, img in enumerate(tqdm(imgs)):
        # interpolate images and labels to the half size. original mapillary images have high expected spatial size, causes memory problems. 
        before_prep = torch.tensor(np.asarray(Image.open(os.path.join(image_dir, img)))).permute(2, 0, 1).unsqueeze(0)/255
        _, _, h, w = before_prep.shape
        after_prep = interp(before_prep, (h//2, w//2), mode="bilinear").squeeze().permute(1, 2, 0).numpy()
        save_image = Image.fromarray((after_prep*255).astype(np.uint8))
        
        
        init_label = np.asarray(Image.open(os.path.join(label_dir, img.split(".")[0] + ".png")))
        transformed_label = 255 * np.ones(init_label.shape, dtype=np.float32)

        for k, v in map_.items():
            transformed_label[init_label == int(k)] = v
        transformed_label = torch.tensor(transformed_label).unsqueeze(0).unsqueeze(0)
        transformed_label = interp(transformed_label, (h//2, w//2), mode="nearest")
        save_label = Image.fromarray(transformed_label.squeeze().numpy().astype(np.int32))

        save_image.save(osp.join(image_save_dir, dataset_name + "_" + str(i) + ".jpg"))
        save_label.save(osp.join(label_save_dir, dataset_name + "_" + str(i) + ".png"))