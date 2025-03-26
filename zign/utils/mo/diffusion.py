import torch
from tqdm import tqdm


class Diffusion:
    
    def __init__(self, noise_steps, beta_start, beta_end, device):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = device
        
        self.beta = self.prepare_noise_schedule().to(device)    # [1000] β_t = β_start + (β_end - β_start)*t/T
        self.alpha = 1. - self.beta # [1000] α_t = 1 - β_t
        self.alpha_hat = torch.cumprod(self.alpha, dim=0) # [1000] α_hat_t = ∏_{i=1}^{t}α_i = α_1*α_2*...*α_t
        
    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
    
    def get_noise_beta(self, t):
        return self.beta[t][:, None, None, None]

    def noise_images(self, x, t):   # x_t = alpha_hat_t**0.5*x_0 + (1 - alpha_hat_t)**0.5*Ɛ
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None] # [batch_size, 1, 1, 1]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x) # 纯噪声
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ
    
    def denoise_images(self, x_t, t, predicted_noise):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None] # [batch_size, 1, 1, 1]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        return (x_t - sqrt_one_minus_alpha_hat * predicted_noise) / sqrt_alpha_hat
    
    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,)).to(self.device)
    
    def sample(self, predict_fn, x, *args):
        for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
            t = (torch.ones(x.size()[0]) * i).long().to(self.device)
            predicted_noise = predict_fn(x, t, *args)
            alpha = self.alpha[t][:, None, None, None]
            alpha_hat = self.alpha_hat[t][:, None, None, None]
            beta = self.beta[t][:, None, None, None]
            if i > 1:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
            x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        return x