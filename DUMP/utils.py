import torch
import numpy as np
from torchdiffeq import odeint_adjoint


def check_grids_alignment(grid1: torch.Tensor, grid2: torch.Tensor, tolerance: float = 1e-6):
    """
    Check if all values in grid2 exist in grid1 within tolerance.

    Args:
        grid1: Reference grid (e.g., solver_z)
        grid2: Grid to check (e.g., target_z)
        tolerance: Maximum allowed difference

    Returns:
        bool: True if all grid2 values are in grid1

    Raises:
        ValueError: If grids are not aligned
    """
    for val in grid2:
        if not torch.any(torch.abs(grid1 - val) < tolerance):
            raise ValueError(
                f"Grid alignment failed: value {val:.6f} from grid2 "
                f"not found in grid1 (tolerance={tolerance})"
            )
    return True

def find_solver_grid(target_z, dz, method="rk4"):
    '''
    Given a target grid, this function quickly runs a small solver
    to get a set of requested steps by torchdiffeq and returns them to the user

    I tried to mimic the usage of rk4 by neural ODE
    '''
    requested_z = []

    # Create a dummy nn.Module to match the model's setup with parameters
    class DummyRHS(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy_param = torch.nn.Parameter(torch.zeros(1))

        def forward(self, z, y):
            requested_z.append(z.item())
            return torch.zeros_like(y) * self.dummy_param  # Use param to create grad graph

    rhs = DummyRHS()
    target_z = torch.as_tensor(target_z)

    # Use odeint_adjoint with parameters to match what the model uses
    solution = odeint_adjoint(
        rhs, y0=torch.zeros(1), t=target_z,
        method=method, options={"step_size": dz},
        adjoint_params=tuple(rhs.parameters())
    )

    # torch calls the rhs multiple times with the same time
    requested_z = np.unique(requested_z)
    #print(f"Captured {len(requested_z)} unique redshift points")
    #print(f"Range: [{requested_z.min():.6f}, {requested_z.max():.6f}]")

    # numpy unique sorts the array, we solve from high z to small z
    return requested_z[::-1]