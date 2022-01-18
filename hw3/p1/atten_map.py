import torch
import torchvision.transforms as transforms
import parser
from pytorch_pretrained_vit import ViT
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

if __name__ == '__main__':
    args = parser.arg_parse()
    ckpt = "../ckpts/log/94.pth"

    # load model and weights
    state = torch.load(ckpt, map_location=torch.device('cpu'))
    model = ViT('B_16', num_classes=37, image_size=args.size, pretrained=True)
    model.load_state_dict(state['state_dict'])

    # transform
    transform = transforms.Compose([
    transforms.Resize((args.size, args.size)),
    transforms.ToTensor(),
    transforms.Normalize( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
    ),
    ])

    # read images
    img_lst = []
    img = Image.open("../hw3_data/p1_data/val/26_5064.jpg")
    img_lst.append(cv2.imread("../hw3_data/p1_data/val/26_5064.jpg"))
    img_lst.append(cv2.imread("../hw3_data/p1_data/val/29_4718.jpg"))
    img_lst.append(cv2.imread("../hw3_data/p1_data/val/31_4838.jpg"))

    print(model)
    print("==================")
    img = transform(img).unsqueeze(0)
    att_mat = model(img)
    # print(att_mat)
    # print(att_mat.shape)
    # # att_mat = torch.stack(att_mat).squeeze(1)
    # att_mat = torch.mean(att_mat, dim=1)
    # print(att_mat)
   
    atten = model.transformer.blocks[11].attn
    # print(atten)
    patches = model.patch_embedding(img) 
    # print("Image tensor: ", img.shape) # [1, 3, 224, 224]
    # print("Patch embeddings: ", patches.shape) #  [1, 768, 14, 14]
    pos_embed = model.positional_embedding.pos_embedding
    # print("Posittion embedding: ", pos_embed.shape)  # [1, 197, 768]
    # model.transformer.
    print(model.class_token.shape) # [1, 1, 768]
    print(patches.shape) #[1, 768, 14, 14]
    transformer_input = torch.cat((model.class_token, patches), dim=1) + pos_embed
    print("Transformer input: ", transformer_input.shape)

    # print(atten.proj_q)
    # print(atten.proj_k)
    # print(atten.proj_v)
    # print(atten)
    def get_attention_map(img, get_mask=False):
        x = transform(img)
        x.size()

        att_mat = model(x.unsqueeze(0))

        # att_mat = torch.stack(att_mat).squeeze(1)

        # Average the attention weights across all heads.
        att_mat = torch.mean(att_mat, dim=1)

        # To account for residual connections, we add an identity matrix to the
        # attention matrix and re-normalize the weights.
        residual_att = torch.eye(att_mat.size(1))
        aug_att_mat = att_mat + residual_att
        aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

        # Recursively multiply the weight matrices
        joint_attentions = torch.zeros(aug_att_mat.size())
        joint_attentions[0] = aug_att_mat[0]

        for n in range(1, aug_att_mat.size(0)):
            joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])

        v = joint_attentions[-1]
        grid_size = int(np.sqrt(aug_att_mat.size(-1)))
        mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()
        if get_mask:
            result = cv2.resize(mask / mask.max(), img.size)
        else:        
            mask = cv2.resize(mask / mask.max(), img.size)[..., np.newaxis]
            result = (mask * img).astype("uint8")

        return  result

def plot_attention_map(original_img, att_map):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 16))
    ax1.set_title('Original')
    ax2.set_title('Attention Map Last Layer')
    ax1.imshow(original_img)
    ax2.imshow(att_map)
    

