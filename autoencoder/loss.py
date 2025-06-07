import torch


class Loss(torch.nn.Module):
    def __init__(
        self,
        lambda_1: float,
        lambda_2: float,
        lambda_3: float,
        lambda_r: float,
        order: int = 1,
        eps: float = 1e-8,
        *args,
        **kwargs
    ) -> None:
        """
        Custom loss function based on multiple normalized MSEs.

        Args:
            lambda_1 (float): loss weight for decoder reconstruction (x vs x_decode)
            lambda_2 (float): loss weight for SINDy latent prediction (dz/ddz vs predicted)
            lambda_3 (float): loss weight for SINDy x/ dx reconstruction
            lambda_r (float): loss weight for SINDy coefficient regularization
            order (int): Order of the model: 1 or 2 (defaults to 1)
            eps (float): small value to avoid division by zero
        """
        super().__init__(*args, **kwargs)
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.lambda_r = lambda_r
        self.eps = eps
        self.regularization = True

        self.reconstruction_loss = 0
        self.latent_loss = 0
        self.sindy_loss = 0

        if order == 1:
            self.forward = self.forward_dx
        else:
            self.forward = self.forward_ddx

    def forward_dx(
        self,
        x,
        dx,
        dz,
        dz_pred,
        x_decode,
        dx_decode,
        sindy_coeffs: torch.Tensor,
        coeff_mask,
    ) -> torch.Tensor:

        loss = 0

        loss += (
            self.lambda_1
            * torch.mean((x - x_decode) ** 2)
            / (torch.mean(x**2) + self.eps)
        )
        loss += (
            self.lambda_2
            * torch.mean((dz - dz_pred) ** 2)
            / (torch.mean(dz**2) + self.eps)
        )
        loss += (
            self.lambda_3
            * torch.mean((dx - dx_decode) ** 2)
            / (torch.mean(dx**2) + self.eps)
        )
        loss += (
            int(self.regularization)
            * self.lambda_r
            * torch.mean(torch.abs(sindy_coeffs) * coeff_mask)
        )

        return loss

    def forward_ddx(
        self,
        x,
        dx,
        dz,
        dz_pred,
        x_decode,
        dx_decode,
        sindy_coeffs: torch.Tensor,
        ddz,
        ddx,
        ddx_decode,
        coeff_mask,
    ) -> torch.Tensor:

        loss = 0

        self.reconstruction_loss = (
            self.lambda_1
            * torch.mean((x - x_decode) ** 2)
            / (torch.mean(x**2) + self.eps)
        )
        self.latent_loss = (
            self.lambda_2
            * torch.mean((dz - dz_pred) ** 2)
            / (torch.mean(dz**2) + self.eps)
        )
        self.sindy_loss = (
            self.lambda_3
            * torch.mean((ddx - ddx_decode) ** 2)
            / (torch.mean(ddx**2) + self.eps)
        )

        loss += self.reconstruction_loss
        # dz_pred is interpreted as ddz_pred in second-order case
        loss += self.latent_loss
        loss += self.sindy_loss
        loss += (
            int(self.regularization)
            * self.lambda_r
            * torch.mean(torch.abs(sindy_coeffs) * coeff_mask)
        )

        return loss

    def set_regularization(self, include_regularization: bool) -> None:
        self.regularization = include_regularization
