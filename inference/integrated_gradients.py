import torch
import numpy as np
from tqdm.auto import tqdm

import glob, os, copy
import PIL
import matplotlib.cm as mpl_color_map

import torchvision.transforms.functional as F
from torchvision.transforms import Resize


def convert_to_grayscale(im_as_arr):
    """
        Converts 3d image to grayscale

    Args:
        im_as_arr (numpy arr): RGB image with shape (D,W,H)

    returns:
        grayscale_im (numpy_arr): Grayscale image with shape (1,W,D)
    """
    grayscale_im = np.sum(np.abs(im_as_arr), axis=0)
    im_max = np.percentile(grayscale_im, 99)
    im_min = np.min(grayscale_im)
    grayscale_im = (np.clip((grayscale_im - im_min) / (im_max - im_min), 0, 1))
    grayscale_im = np.expand_dims(grayscale_im, axis=0)
    return grayscale_im


def apply_colormap_on_image(org_im,
                            activation,
                            colormap_name,
                            heatmap_alpha=0.2):
    """
        Apply heatmap on image
    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): Activation map (grayscale) 0-255
        colormap_name (str): Name of the colormap
    """
    # Get colormap
    color_map = mpl_color_map.get_cmap(colormap_name)
    no_trans_heatmap = color_map(activation)
    no_trans_heatmap = no_trans_heatmap
    no_trans_heatmap = np.squeeze(no_trans_heatmap, axis=2)
    print("no_trans_heatmap", no_trans_heatmap.shape)
    # Change alpha channel in colormap to make sure original image is displayed
    heatmap = copy.copy(no_trans_heatmap)
    heatmap[:, :, 3] = heatmap_alpha
    heatmap = PIL.Image.fromarray((heatmap * 255).astype(np.uint8))
    no_trans_heatmap = PIL.Image.fromarray(
        (no_trans_heatmap * 255).astype(np.uint8))

    # Apply heatmap on image
    heatmap_on_image = PIL.Image.new("RGBA", org_im.shape[:2])
    heatmap_on_image = PIL.Image.alpha_composite(
        heatmap_on_image,
        PIL.Image.fromarray(org_im).convert('RGBA'))
    heatmap_on_image = PIL.Image.alpha_composite(heatmap_on_image, heatmap)
    return no_trans_heatmap, heatmap_on_image


def format_np_output(np_arr):
    if len(np_arr.shape) == 2:
        np_arr = np.expand_dims(np_arr, axis=0)
    if np_arr.shape[0] == 1:
        np_arr = np.repeat(np_arr, 3, axis=0)
    if np_arr.shape[0] == 3:
        np_arr = np_arr.transpose(1, 2, 0)
    if np.max(np_arr) <= 1:
        np_arr = (np_arr * 255).astype(np.uint8)
    return np_arr


def save_image(im, path):
    if isinstance(im, (np.ndarray, np.generic)):
        im = format_np_output(im)
        im = PIL.Image.fromarray(im)
    im.save(path)


def save_gradient_images(gradient, file_path):
    gradient = gradient - gradient.min()
    gradient /= gradient.max()
    # Save image
    save_image(gradient, file_path)


class IntegratedGradients():

    def __init__(self, model):
        self.model = model
        self.gradients = None
        # Put model in evaluation mode
        self.model.eval()
        # Hook the first layer to get the gradient

    def generate_images_on_linear_path(self, input_image, steps):
        # Generate uniform numbers between 0 and steps
        step_list = np.arange(steps + 1) / steps
        # Generate scaled xbar images
        xbar_list = [input_image * step for step in step_list]
        return xbar_list

    def generate_gradients(self, input_image, target_class=None):
        # Forward
        input_image.requires_grad_()
        self.model.eval()
        model_output = self.model(input_image)
        # Zero grads
        self.model.zero_grad()
        # Target for backprop
        model_output.backward(torch.ones_like(model_output))
        gradients = input_image.grad.data.abs()
        return gradients.cpu().detach().numpy()

    def generate_integrated_gradients(self,
                                      input_image,
                                      num_steps,
                                      target_class=None):
        xbar_list = self.generate_images_on_linear_path(input_image, num_steps)
        integrated_grads = np.zeros(input_image.size())
        for xbar_image in xbar_list:
            single_integrated_grad = self.generate_gradients(
                xbar_image, target_class)
            integrated_grads = integrated_grads + single_integrated_grad / num_steps
        return integrated_grads[0]


def integrate_gradients(model,
                        image_size=512,
                        num_steps=50,
                        path="test_images/tweets/"):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_files = glob.glob(path)
    image_files = [f for f in image_files if os.path.isfile(f)]
    for _, image_path in enumerate(tqdm(image_files, total=len(image_files))):
        image = PIL.Image.open(image_path).convert("RGB")
        image = Resize((image_size, image_size))(image)
        image_name = image_path.split("/")[-1].split(".")[0]
        image.save(f"./features/{image_name}.png")
        image = F.to_tensor(image).to(device).unsqueeze(0)
        print(image_path)
        preds = model(image)[0]
        class_idx = preds.argmax(-1).item()
        preds_dist = preds.softmax(-1).cpu().detach().tolist()
        preds_dist = ['{:.2f}'.format(x) for x in preds_dist]
        print(image_path, preds.softmax(-1), class_idx)
        IG = IntegratedGradients(model)
        integrated_grads = IG.generate_integrated_gradients(image, num_steps)
        integrated_grads = convert_to_grayscale(integrated_grads)
        try:
            image = image[0].cpu().detach().numpy().transpose(1, 2, 0)
            integrated_grads = integrated_grads.transpose(1, 2, 0)
            heatmap, integrated_grads = apply_colormap_on_image(
                (image * 255).astype(np.uint8), integrated_grads, None, 0.8)

            integrated_grads.save(
                f"./features/{image_name}_class_{class_idx}_{str(preds_dist)}.png"
            )
        except:
            print("FAILED", image_name)
            continue
