import torch
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import comfy.model_management as model_management

class NewColorNode:
    """
    ComfyUI颜色调整节点 - NewColor
    支持多种颜色调整模式和RGB调整
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "brightness": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "contrast": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "saturation": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "hue": ("FLOAT", {"default": 0.0, "min": -0.5, "max": 0.5, "step": 0.01}),
                "rgb_red": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "rgb_green": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "rgb_blue": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "exposure": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01}),
                "gamma": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.01}),
                "vibrance": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "shadows": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "highlights": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_color_adjustment"
    CATEGORY = "image/color"
    
    def apply_color_adjustment(self, image, brightness, contrast, saturation, hue, 
                                rgb_red, rgb_green, rgb_blue, exposure, gamma, 
                                vibrance, shadows, highlights, mask=None):
        """
        应用颜色调整
        
        Args:
            image: 输入的图像张量 (batch, height, width, channels)
            mask: 可选的遮罩张量 (batch, height, width) 或 (height, width)
            其他参数: 各种颜色调整参数
            
        Returns:
            调整后的图像张量
        """
        # 获取设备
        device = model_management.get_torch_device()
        
        # 将图像转换为numpy数组
        batch_size = image.shape[0]
        results = []
        
        for i in range(batch_size):
            img = image[i].cpu().numpy()
            
            # 转换为0-255范围的uint8
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
            
            # 转换为PIL Image
            pil_image = Image.fromarray(img, mode='RGB')
            
            # 保存原始图像（用于遮罩混合）
            original_image = pil_image.copy()
            
            # 应用颜色调整
            adjusted_image = self._apply_adjustments(
                pil_image, brightness, contrast, saturation, hue,
                rgb_red, rgb_green, rgb_blue, exposure, gamma,
                vibrance, shadows, highlights
            )
            
            # 如果有遮罩，应用遮罩（混合原始图像和调整后的图像）
            if mask is not None:
                adjusted_image = self._apply_mask(
                    original_image, adjusted_image, mask, img.shape[:2], i
                )
            
            # 转换回张量格式
            img_array = np.array(adjusted_image).astype(np.float32) / 255.0
            results.append(img_array)
        
        # 组合成批次张量
        result_tensor = torch.from_numpy(np.stack(results)).to(device)
        
        return (result_tensor,)
    
    def _apply_adjustments(self, pil_image, brightness, contrast, saturation, hue,
                           rgb_red, rgb_green, rgb_blue, exposure, gamma,
                           vibrance, shadows, highlights):
        """
        应用所有颜色调整
        """
        img = pil_image.copy()
        
        # 1. 亮度调整
        if brightness != 1.0:
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(brightness)
        
        # 2. 对比度调整
        if contrast != 1.0:
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(contrast)
        
        # 3. 饱和度调整
        if saturation != 1.0:
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(saturation)
        
        # 4. 色相调整
        if hue != 0.0:
            img = self._adjust_hue(img, hue)
        
        # 5. RGB通道调整
        if rgb_red != 0.0 or rgb_green != 0.0 or rgb_blue != 0.0:
            img = self._adjust_rgb(img, rgb_red, rgb_green, rgb_blue)
        
        # 6. 曝光调整
        if exposure != 0.0:
            img = self._adjust_exposure(img, exposure)
        
        # 7. Gamma调整
        if gamma != 1.0:
            img = self._adjust_gamma(img, gamma)
        
        # 8. 自然饱和度（Vibrance）调整
        if vibrance != 0.0:
            img = self._adjust_vibrance(img, vibrance)
        
        # 9. 阴影调整
        if shadows != 0.0:
            img = self._adjust_shadows(img, shadows)
        
        # 10. 高光调整
        if highlights != 0.0:
            img = self._adjust_highlights(img, highlights)
        
        return img
    
    def _adjust_hue(self, img, hue_shift):
        """
        调整色相
        hue_shift: -0.5 到 0.5 之间
        """
        img_array = np.array(img).astype(np.float32)
        hsv = self._rgb_to_hsv(img_array)
        
        # 调整色相（HSV中的H通道）
        hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 1.0
        
        rgb = self._hsv_to_rgb(hsv)
        return Image.fromarray(rgb.astype(np.uint8), mode='RGB')
    
    def _adjust_rgb(self, img, red, green, blue):
        """
        调整RGB各通道
        red, green, blue: -1.0 到 1.0 之间
        """
        img_array = np.array(img).astype(np.float32)
        
        # 将调整值映射到乘数
        red_mult = 1.0 + red
        green_mult = 1.0 + green
        blue_mult = 1.0 + blue
        
        # 应用RGB调整
        img_array[:, :, 0] = np.clip(img_array[:, :, 0] * red_mult, 0, 255)
        img_array[:, :, 1] = np.clip(img_array[:, :, 1] * green_mult, 0, 255)
        img_array[:, :, 2] = np.clip(img_array[:, :, 2] * blue_mult, 0, 255)
        
        return Image.fromarray(img_array.astype(np.uint8), mode='RGB')
    
    def _adjust_exposure(self, img, exposure):
        """
        调整曝光
        exposure: -2.0 到 2.0 之间
        """
        img_array = np.array(img).astype(np.float32)
        
        # 曝光调整：2^exposure
        exposure_factor = 2.0 ** exposure
        img_array = img_array * exposure_factor
        
        img_array = np.clip(img_array, 0, 255)
        return Image.fromarray(img_array.astype(np.uint8), mode='RGB')
    
    def _adjust_gamma(self, img, gamma):
        """
        调整Gamma
        gamma: 0.1 到 3.0 之间
        """
        img_array = np.array(img).astype(np.float32) / 255.0
        
        # Gamma校正
        img_array = np.power(img_array, 1.0 / gamma)
        
        img_array = (img_array * 255).astype(np.uint8)
        return Image.fromarray(img_array, mode='RGB')
    
    def _adjust_vibrance(self, img, vibrance):
        """
        调整自然饱和度（Vibrance）
        vibrance: -1.0 到 1.0 之间
        """
        img_array = np.array(img).astype(np.float32)
        hsv = self._rgb_to_hsv(img_array)
        
        # 只调整低饱和度的区域
        saturation = hsv[:, :, 1]
        # 计算调整系数（低饱和度区域调整更多）
        adjustment = 1.0 + vibrance * (1.0 - saturation)
        saturation = np.clip(saturation * adjustment, 0, 1.0)
        
        hsv[:, :, 1] = saturation
        rgb = self._hsv_to_rgb(hsv)
        return Image.fromarray(rgb.astype(np.uint8), mode='RGB')
    
    def _adjust_shadows(self, img, shadows):
        """
        调整阴影
        shadows: -1.0 到 1.0 之间
        """
        img_array = np.array(img).astype(np.float32) / 255.0
        
        # 计算亮度
        gray = np.dot(img_array, [0.299, 0.587, 0.114])
        
        # 创建阴影遮罩（暗部）
        shadow_mask = np.clip(1.0 - gray, 0, 1.0)
        
        # 应用阴影调整
        adjustment = 1.0 + shadows * shadow_mask
        img_array = img_array * adjustment[:, :, np.newaxis]
        
        img_array = np.clip(img_array * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(img_array, mode='RGB')
    
    def _adjust_highlights(self, img, highlights):
        """
        调整高光
        highlights: -1.0 到 1.0 之间
        """
        img_array = np.array(img).astype(np.float32) / 255.0
        
        # 计算亮度
        gray = np.dot(img_array, [0.299, 0.587, 0.114])
        
        # 创建高光遮罩（亮部）
        highlight_mask = gray
        
        # 应用高光调整
        adjustment = 1.0 + highlights * highlight_mask
        img_array = img_array * adjustment[:, :, np.newaxis]
        
        img_array = np.clip(img_array * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(img_array, mode='RGB')
    
    def _apply_mask(self, original_image, adjusted_image, mask, img_shape, batch_idx):
        """
        应用遮罩到调整后的图像
        
        Args:
            original_image: 原始PIL图像
            adjusted_image: 调整后的PIL图像
            mask: 遮罩张量
            img_shape: 原始图像形状 (height, width)
            batch_idx: 批次索引
            
        Returns:
            应用遮罩后的PIL图像（仅在遮罩区域内应用调整）
        """
        # 获取遮罩
        if len(mask.shape) == 3:  # (batch, height, width)
            mask_array = mask[batch_idx].cpu().numpy()
        else:  # (height, width)
            mask_array = mask.cpu().numpy()
        
        # 确保遮罩尺寸匹配
        if mask_array.shape != img_shape:
            # 调整遮罩大小
            mask_pil = Image.fromarray((mask_array * 255).astype(np.uint8), mode='L')
            mask_pil = mask_pil.resize(adjusted_image.size, Image.LANCZOS)
            mask_array = np.array(mask_pil).astype(np.float32) / 255.0
        else:
            # 调整遮罩大小到图像尺寸（以防万一）
            mask_pil = Image.fromarray((mask_array * 255).astype(np.uint8), mode='L')
            mask_pil = mask_pil.resize(adjusted_image.size, Image.LANCZOS)
            mask_array = np.array(mask_pil).astype(np.float32) / 255.0
        
        # 如果遮罩值不在0-1范围，归一化
        if mask_array.max() > 1.0:
            mask_array = mask_array / mask_array.max()
        
        # 将原始图像和调整后的图像转换为数组
        original_array = np.array(original_image).astype(np.float32)
        adjusted_array = np.array(adjusted_image).astype(np.float32)
        
        # 使用遮罩混合图像
        # 遮罩区域（白色/高值）使用调整后的图像
        # 非遮罩区域（黑色/低值）使用原始图像
        mask_3d = mask_array[:, :, np.newaxis]  # 扩展为3通道
        
        # 混合：mask区域显示adjusted，非mask区域显示original
        blended_array = original_array * (1.0 - mask_3d) + adjusted_array * mask_3d
        
        # 转换回PIL图像
        blended_image = Image.fromarray(blended_array.astype(np.uint8), mode='RGB')
        
        return blended_image
    
    def _rgb_to_hsv(self, rgb):
        """
        将RGB转换为HSV
        rgb: shape (height, width, 3), 值范围 0-255 或 0-1
        """
        rgb = rgb.astype(np.float32)
        if rgb.max() > 1.0:
            rgb = rgb / 255.0
        
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        
        max_val = np.maximum(np.maximum(r, g), b)
        min_val = np.minimum(np.minimum(r, g), b)
        delta = max_val - min_val
        
        # V (亮度)
        v = max_val
        
        # S (饱和度)
        s = np.zeros_like(v)
        s[max_val != 0] = delta[max_val != 0] / max_val[max_val != 0]
        
        # H (色相)
        h = np.zeros_like(v)
        
        # 当delta不为0时计算色相
        mask = delta != 0
        
        # 红色是最大值
        r_mask = mask & (max_val == r)
        h[r_mask] = (((g[r_mask] - b[r_mask]) / delta[r_mask]) % 6) / 6.0
        
        # 绿色是最大值
        g_mask = mask & (max_val == g)
        h[g_mask] = (((b[g_mask] - r[g_mask]) / delta[g_mask]) + 2) / 6.0
        
        # 蓝色是最大值
        b_mask = mask & (max_val == b)
        h[b_mask] = (((r[b_mask] - g[b_mask]) / delta[b_mask]) + 4) / 6.0
        
        hsv = np.stack([h, s, v], axis=2)
        return hsv
    
    def _hsv_to_rgb(self, hsv):
        """
        将HSV转换为RGB
        hsv: shape (height, width, 3), 值范围 0-1
        """
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        
        # 将色相转换为0-360度
        h = (h * 6.0) % 6.0
        
        c = v * s
        x = c * (1 - np.abs((h % 2) - 1))
        m = v - c
        
        r, g, b = np.zeros_like(h), np.zeros_like(h), np.zeros_like(h)
        
        # 根据色相区间设置RGB值
        mask0 = (h >= 0) & (h < 1)
        mask1 = (h >= 1) & (h < 2)
        mask2 = (h >= 2) & (h < 3)
        mask3 = (h >= 3) & (h < 4)
        mask4 = (h >= 4) & (h < 5)
        mask5 = (h >= 5) & (h < 6)
        
        r[mask0] = c[mask0]
        g[mask0] = x[mask0]
        b[mask0] = 0
        
        r[mask1] = x[mask1]
        g[mask1] = c[mask1]
        b[mask1] = 0
        
        r[mask2] = 0
        g[mask2] = c[mask2]
        b[mask2] = x[mask2]
        
        r[mask3] = 0
        g[mask3] = x[mask3]
        b[mask3] = c[mask3]
        
        r[mask4] = x[mask4]
        g[mask4] = 0
        b[mask4] = c[mask4]
        
        r[mask5] = c[mask5]
        g[mask5] = 0
        b[mask5] = x[mask5]
        
        rgb = np.stack([r + m, g + m, b + m], axis=2)
        rgb = np.clip(rgb * 255, 0, 255)
        return rgb

