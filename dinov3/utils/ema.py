"""
Exponential Moving Average (EMA) Teacher Model
Used for generating pseudo-labels in HRDA self-training.

Based on HRDA paper Eq. 5: φ_{t+1} ← α φ_t + (1 - α) θ_t
"""

import torch
import torch.nn as nn
from copy import deepcopy


class EMATeacher(nn.Module):
    """
    Exponential Moving Average Teacher Model for Self-Training.
    
    Maintains a temporally smoothed version of the student model to generate
    more stable pseudo-labels.
    
    Args:
        model: Student model to create EMA from
        alpha: EMA momentum (default: 0.999 as per HRDA paper)
    """
    
    def __init__(self, model, alpha=0.999):
        super(EMATeacher, self).__init__()
        
        self.alpha = alpha
        self.global_step = 0
        
        # Create EMA model as a deep copy
        self.ema_model = deepcopy(model)
        
        # Disable gradient computation for EMA model
        for param in self.ema_model.parameters():
            param.requires_grad = False
        
        # Set to eval mode
        self.ema_model.eval()
        
        print(f"EMA Teacher initialized with alpha={alpha}")
    
    def update(self, model):
        """
        Update EMA model parameters.
        
        φ_{t+1} ← α φ_t + (1 - α) θ_t
        
        Args:
            model: Current student model
        """
        self.global_step += 1
        
        # EMA update
        with torch.no_grad():
            for ema_param, model_param in zip(
                self.ema_model.parameters(), 
                model.parameters()
            ):
                ema_param.data.mul_(self.alpha).add_(
                    model_param.data, alpha=1 - self.alpha
                )
    
    def forward(self, *args, **kwargs):
        """Forward pass through EMA model"""
        return self.ema_model(*args, **kwargs)
    
    def state_dict(self):
        """Get state dict including EMA model"""
        return {
            'ema_model': self.ema_model.state_dict(),
            'alpha': self.alpha,
            'global_step': self.global_step
        }
    
    def load_state_dict(self, state_dict):
        """Load state dict"""
        self.ema_model.load_state_dict(state_dict['ema_model'])
        self.alpha = state_dict.get('alpha', self.alpha)
        self.global_step = state_dict.get('global_step', 0)
    
    def eval(self):
        """Set to evaluation mode"""
        self.ema_model.eval()
        return self
    
    def train(self, mode=True):
        """EMA model always stays in eval mode"""
        self.ema_model.eval()
        return self


if __name__ == "__main__":
    # Test EMA teacher
    from models.mamnet import MAMNet
    
    # Create student model
    student = MAMNet(num_classes=2, pretrained=False)
    
    # Create EMA teacher
    teacher = EMATeacher(student, alpha=0.999)
    
    print(f"Student parameters: {sum(p.numel() for p in student.parameters()):,}")
    print(f"Teacher parameters: {sum(p.numel() for p in teacher.ema_model.parameters()):,}")
    
    # Simulate training step
    x = torch.randn(2, 3, 384, 384)
    
    # Student forward
    student.train()
    student_out = student(x)
    print(f"Student output shape: {student_out['main'].shape if isinstance(student_out, dict) else student_out.shape}")
    
    # Teacher forward (no gradients)
    teacher.eval()
    with torch.no_grad():
        teacher_out = teacher(x)
    print(f"Teacher output shape: {teacher_out['main'].shape if isinstance(teacher_out, dict) else teacher_out.shape}")
    
    # Update teacher
    teacher.update(student)
    print(f"Teacher updated (step {teacher.global_step})")