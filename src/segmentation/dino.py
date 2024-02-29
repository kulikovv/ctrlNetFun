import torch
import torchvision.transforms as T
import PIL


class DinoSegmentation:
    def __init__(self, device: str = "cuda") -> None:
        # we will use small dino for speed and we don't need high accuracy
        self.dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        self.feat_dim = 384

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.dinov2.to(self.device)

        self.patch_h = 40
        self.patch_w = 40
        self.transform = T.Compose([
            T.Resize((self.patch_h * 14, self.patch_w * 14)),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    @torch.no_grad()
    def __call__(self,
                 image: PIL.Image.Image,
                 threshold: float = 1e-3) -> torch.FloatTensor:

        imgs_tensor = self.transform(image)[:3].unsqueeze(0).to(self.device)

        features_dict = self.dinov2.forward_features(imgs_tensor)
        features = features_dict['x_norm_patchtokens']
        features = features.reshape(self.patch_h * self.patch_w, self.feat_dim)

        u, _, _ = torch.pca_lowrank(features, q=3)

        # segment using the first component
        pca_features_fg = u[:, 0] > threshold

        return pca_features_fg.reshape(1, 1, self.patch_h, self.patch_w).cpu()


if __name__ == "__main__":
    from diffusers.utils import load_image  # ignore

    original_image = load_image(
        "https://cdn.pixabay.com/photo/2017/02/20/18/03/cat-2083492_640.jpg"
    )

    md = DinoSegmentation()

    matte = md(original_image, 0.01)

    im = PIL.Image.fromarray(((matte[0, 0].numpy() * 255).astype('uint8')), mode='L')
    im = im.resize((512, 512), resample=PIL.Image.LANCZOS)
    im.save("dino_output.png")
