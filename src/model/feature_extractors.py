import torch
import torch.nn.functional as F
from .samplers import ApproximateGreedyCoresetSampler
from .utils import load_backbone
class PatchFeatureExtractor(torch.nn.Module):
    def __init__(
        self, 
        device="cuda", 
        backbone_name="wideresnet50", 
        flatten_dimension=1024,
        out_dimension = 1024,
        patchsize=3,
        patchstride=1,
        sampler = ApproximateGreedyCoresetSampler(0.2, "cuda"),
    ):
        super(PatchFeatureExtractor, self).__init__()
        self.backbone = load_backbone(backbone_name)
        self.outputs = {}
        self.device = device
        self.flatten_dimension = flatten_dimension
        self.out_dimension = out_dimension
        self.patchsize = patchsize
        self.patchstride = patchstride
        self.padding = int((patchsize - 1) / 2)
        self.sampler = sampler
        self.to(device)
        if not hasattr(self.backbone, "hook_handles"):
            self.backbone.hook_handles = []
        for handle in self.backbone.hook_handles:
            handle.remove()

        for extract_layer, forward_hook in zip(["layer2", "layer3"], [self._forward_hook_layer2, self._forward_hook_layer3]):

            network_layer = self.backbone.__dict__["_modules"][extract_layer]
            if isinstance(network_layer, torch.nn.Sequential):
                self.backbone.hook_handles.append(
                    network_layer[-1].register_forward_hook(forward_hook)
                )
            else:
                self.backbone.hook_handles.append(
                    network_layer.register_forward_hook(forward_hook)
                )
        self.eval()
                
    def _forward_hook_layer2(self, module, input, output):
        self.outputs["layer2"] = output
    def _forward_hook_layer3(self, module, input, output):
        self.outputs["layer3"] = output
        
    def forward(self, images):
        self.outputs.clear()
        with torch.no_grad():
            try:
                _ = self.backbone(images)
                features = []
                for output_index, (layer_name, _features) in enumerate(self.outputs.items()):                    
                    # Patchfy: B x C x H x W --> B x C*Patchsize*Patchsize x (H-stride+1)*(W-stride+1)
                    # result : B x C x H x W  --> B*PatchDim*PatchDim  x  flatten_dimension
                    unfolder = torch.nn.Unfold(
                        kernel_size=self.patchsize, stride=self.patchstride, padding=self.padding, dilation=1
                    )
                    unfolded_features = unfolder(_features)
                    number_of_total_patches = []
                    for s in _features.shape[-2:]:
                        n_patches = (
                            s + 2 * self.padding - 1 * (self.patchsize - 1) - 1
                        ) / self.patchstride + 1
                        number_of_total_patches.append(int(n_patches))
                    unfolded_features = unfolded_features.reshape(
                        *_features.shape[:2], self.patchsize, self.patchsize, -1
                    )

                    _features = unfolded_features.permute(0, 4, 1, 2, 3)

                    patch_dims = number_of_total_patches
                    if output_index == 0:
                        ref_num_patches = patch_dims

                    _features = _features.reshape(
                        _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
                    )
                    _features = _features.permute(0, -3, -2, -1, 1, 2)
                    perm_base_shape = _features.shape
                    _features = _features.reshape(-1, *_features.shape[-2:])
                    _features = F.interpolate(
                        _features.unsqueeze(1),
                        size=(ref_num_patches[0], ref_num_patches[1]),
                        mode="bilinear",
                        align_corners=False,
                    )
                    _features = _features.squeeze(1)
                    _features = _features.reshape(
                        *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
                    )
                    _features = _features.permute(0, -2, -1, 1, 2, 3)
                    _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
                    _features = _features.reshape(-1, *_features.shape[-3:])
                    
                    
                    _features = _features.reshape(len(_features), 1, -1)
                    _features = F.adaptive_avg_pool1d(_features, self.flatten_dimension).squeeze(1)
                    features.append(_features)
                    
                # aggregator: merge 2 layer features
                features = torch.stack(features, dim=1)
                
                features = features.reshape(len(features), 1, -1)
                features = F.adaptive_avg_pool1d(features, self.out_dimension)
                features = torch.squeeze(features, 1)
                
                
            except Exception as e:
                print(e)
        return features, ref_num_patches