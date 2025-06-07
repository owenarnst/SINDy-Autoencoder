from typing import Dict
import torch
from SINDy_library import *
from autoencoder.autoencoder import AutoEncoder
from SINDy_library import library_size


class SINDy(torch.nn.Module):
    """

    Description: Custom neural network module that embeds a SINDy model into an autoencoder.

    """

    def __init__(
        self,
        encoder: AutoEncoder,
        decoder: AutoEncoder,
        device: str,
        params: dict = {},
        *args,
        **kwargs
    ) -> None:
        """

        Description: Constructor for the SINDy class. Initializes the model parameters, encoder, and decoder.

        Args:
            encoder (AutoEncoder): The encoder part of the autoencoder.
            decoder (AutoEncoder): The decoder part of the autoencoder.
            params (Dict): A dictionary containing model and SINDy parameters.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        """

        super().__init__(*args, **kwargs)

        # Initialize model parameters, encoder, and decoder
        self.params = params
        self.encoder = encoder
        self.decoder = decoder

        # Set model order to help in intializing other attributes
        self.model_order = self.params["model_order"]

        # Initialize autoencoder parameters ----------
        self.input_dim = self.params["input_dim"]
        self.latent_dim = self.params["latent_dim"]

        # Initialize SINDy parameters ------------------------------------------------------

        # Library parameters
        self.poly_order = self.params["poly_order"]
        self.include_sine = self.params["include_sine"]
        if self.model_order == 1:
            self.library_dim = library_size(
                self.params["latent_dim"],
                self.params["poly_order"],
                self.params["include_sine"],
            )
        elif self.model_order == 2:
            self.library_dim = library_size(
                2 * self.params["latent_dim"],
                self.params["poly_order"],
                self.params["include_sine"],
            )

        # Coefficient parameters
        self.sequential_thresholding = self.params["sequential_thresholding"]
        self.coefficient_initialization = self.params["coefficient_initialization"]
        self.coefficient_mask = torch.ones((self.library_dim, self.latent_dim)).to(
            device
        )
        self.coefficient_threshold = self.params["coefficient_threshold"]

        # Greek letter 'Xi' in the paper. Learned during training (different from linear regression).
        sindy_coefficients = self.init_sindy_coefficients(
            self.params["coefficient_initialization"]
        )
        # Treat sindy_coefficients as a parameter to be learned and move it to device
        self.sindy_coefficients = torch.nn.Parameter(
            sindy_coefficients.to(torch.float64).to(device)
        )

        # Order of dynamical system
        self.model_order = self.params["model_order"]
        if self.model_order == 1:
            self.forward = self.forward_dx
        else:
            self.forward = self.forward_ddx

    def init_sindy_coefficients(self, name="normal", std=1.0, k=1) -> torch.Tensor:
        """

        Description: Initializes the SINDy coefficients based on the specified method. These coefficients are learned during training.

        Args:
            name (str): The method for initializing the coefficients. Options are 'xavier', 'uniform', 'constant', and 'normal'.
            std (float): Standard deviation for normal initialization.
            k (float): Constant value for constant initialization.

        """

        sindy_coefficients = torch.zeros((self.library_dim, self.latent_dim))

        if name == "xavier":
            return torch.nn.init.xavier_uniform_(sindy_coefficients)
        elif name == "uniform":
            return torch.nn.init.uniform_(sindy_coefficients, low=0.0, high=1.0)
        elif name == "constant":
            return torch.ones_like(sindy_coefficients) * k
        elif name == "normal":
            return torch.nn.init.normal_(sindy_coefficients, mean=0, std=std)

    def forward_dx(self, x, dx) -> torch.Tensor:
        """

        Description: Forward pass for the SINDy model with first-order derivatives.

        Args:
            x (torch.Tensor): Input tensor representing the state of the system.
            dx (torch.Tensor): Input tensor representing the first-order derivatives of the state.

        Returns:
            torch.Tensor: The output tensors including the original state, first-order derivatives, predicted derivatives, and decoded states.

        """

        # pass input through encoder
        out_encode = self.encoder(torch.cat((x, dx)))
        dz = out_encode[out_encode.shape[0] // 2 :]
        z = out_encode[: out_encode.shape[0] // 2]

        # create library
        Theta = sindy_library_pt(z, self.latent_dim, self.poly_order, self.include_sine)

        # apply thresholding or not
        if self.sequential_thresholding:
            sindy_predict = torch.matmul(
                Theta, self.coefficient_mask * self.sindy_coefficients
            )
        else:
            sindy_predict = torch.matmul(Theta, self.sindy_coefficients)

        # decode transformed input (z) and predicted derivatives (z dot)
        x_decode = self.decoder(torch.cat((z, sindy_predict)))
        dx_decode = x_decode[x_decode.shape[0] // 2 :]
        x_decode = x_decode[: x_decode.shape[0] // 2]

        dz_predict = sindy_predict

        return (
            x,
            dx,
            dz_predict,
            dz,
            x_decode,
            dx_decode,
            self.sindy_coefficients,
        )

    def forward_ddx(self, x: torch.Tensor, dx: torch.Tensor, ddx: torch.Tensor):
        """

        Description: Forward pass for the SINDy model with second-order derivatives.

        Args:
            x (torch.Tensor): Input tensor representing the state of the system.
            dx (torch.Tensor): Input tensor representing the first-order derivatives of the state.
            ddx (torch.Tensor): Input tensor representing the second-order derivatives of the state.

        """

        out = self.encoder(torch.cat((x, dx, ddx)))
        slicer = out.shape[0] // 3
        z, dz, ddz = out[:slicer], out[slicer : 2 * slicer], out[2 * slicer :]

        # create Theta
        Theta = sindy_library_pt_order2(
            z, dz, self.latent_dim, self.poly_order, self.include_sine
        )

        # apply thresholding or not
        if self.sequential_thresholding:
            sindy_predict = torch.matmul(
                Theta, self.coefficient_mask * self.sindy_coefficients
            )
        else:
            sindy_predict = torch.matmul(Theta, self.sindy_coefficients)

        # decode
        out_decode = self.decoder(torch.cat((z, dz, sindy_predict)))
        slicer = out_decode.shape[0] // 3
        x_decode, dx_decode, ddx_decode = (
            out_decode[:slicer],
            out_decode[slicer : 2 * slicer],
            out_decode[2 * slicer :],
        )

        ddz_predict = sindy_predict

        return (
            x,
            dx,
            dz,
            ddz_predict,
            x_decode,
            dx_decode,
            self.sindy_coefficients,
            ddz,
            ddx,
            ddx_decode,
            self.coefficient_mask,
        )
