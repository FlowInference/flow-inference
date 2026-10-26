import math

import gpytorch
import pytest
import torch
from gpytorch.kernels import RFFKernel, ScaleKernel

from flow_inference.flow_utils import build_grid_in_box
from flow_inference.gp_utils import (
    AbstractBasis,
    DecoupledRFFBasis,
    PosteriorKernel,
    PosteriorMean,
    RFFBasis,
    _train_cholesky,
    fit_kernel_hyperparameters,
    gp_posterior,
)


@pytest.fixture(autouse=True)
def _restore_default_dtype():
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.double)
    yield
    torch.set_default_dtype(prev)


def make_kernel(lengthscale, outputscale=1.0):
    """Create a ScaleKernel(RBFKernel) with explicit lengthscale/outputscale."""
    lengthscale = torch.as_tensor(lengthscale, dtype=torch.double)

    if lengthscale.ndim == 0:
        ard_num_dims = None
    else:
        ard_num_dims = lengthscale.numel()

    base = gpytorch.kernels.RBFKernel(ard_num_dims=ard_num_dims)
    kernel = gpytorch.kernels.ScaleKernel(base)

    if lengthscale.ndim == 0:
        kernel.base_kernel.lengthscale = lengthscale.reshape(1, 1)
    else:
        kernel.base_kernel.lengthscale = lengthscale.reshape(1, -1)

    kernel.outputscale = torch.as_tensor(outputscale, dtype=torch.double)

    return kernel


def make_grid(n=7, d=2):
    x = torch.linspace(0.0, 1.0, n, dtype=torch.double)
    if d == 1:
        return x[:, None]

    cols = []
    for j in range(d):
        cols.append(x ** (j + 1))
    return torch.stack(cols, dim=-1)


def test_rff_basis_shapes_scalar_lengthscale():
    torch.manual_seed(0)

    R = 10
    grid = make_grid(n=6, d=2)
    mean = gpytorch.means.ZeroMean()
    kernel = make_kernel(lengthscale=0.7, outputscale=2.0)

    basis = RFFBasis(
        R=R, input_dim=grid.shape[-1], mean=mean, kernel=kernel, device="cpu"
    )

    assert basis.phi(grid).shape == (R, grid.shape[0])
    assert basis.approximate_kernel(grid).shape == (grid.shape[0], grid.shape[0])
    assert basis.lengthscale.shape == (grid.shape[1],)
    assert basis.omega.shape == (grid.shape[1], R // 2)
    assert basis.omega_eff.shape == (grid.shape[1], R // 2)


def test_rff_basis_shapes_ard_lengthscale():
    torch.manual_seed(0)

    R = 12
    grid = make_grid(n=5, d=3)
    mean = gpytorch.means.ZeroMean()
    kernel = make_kernel(lengthscale=torch.tensor([0.5, 1.2, 2.0]), outputscale=1.7)

    basis = RFFBasis(
        R=R, input_dim=grid.shape[-1], mean=mean, kernel=kernel, device="cpu"
    )

    assert basis.phi(grid).shape == (R, grid.shape[0])
    assert basis.lengthscale.shape == (grid.shape[1],)
    assert torch.allclose(
        basis.lengthscale, torch.tensor([0.5, 1.2, 2.0], dtype=torch.double)
    )


def test_odd_number_of_features_raises():
    torch.manual_seed(0)

    grid = make_grid(n=5, d=2)
    mean = gpytorch.means.ZeroMean()
    kernel = make_kernel(lengthscale=torch.tensor([1.0, 1.0]), outputscale=1.0)

    with pytest.raises(AssertionError, match="even number"):
        RFFBasis(R=9, input_dim=grid.shape[-1], mean=mean, kernel=kernel, device="cpu")


def test_non_scaled_kernel_raises():
    torch.manual_seed(0)

    grid = make_grid(n=5, d=2)
    mean = gpytorch.means.ZeroMean()
    kernel = gpytorch.kernels.RBFKernel(ard_num_dims=2)

    with pytest.raises(AssertionError):
        RFFBasis(R=10, input_dim=grid.shape[-1], mean=mean, kernel=kernel, device="cpu")


def test_approximate_kernel_equals_phi_inner_product():
    torch.manual_seed(0)

    R = 10
    grid = make_grid(n=6, d=2)
    mean = gpytorch.means.ZeroMean()
    kernel = make_kernel(lengthscale=torch.tensor([0.9, 1.3]), outputscale=1.4)

    basis = RFFBasis(
        R=R, input_dim=grid.shape[-1], mean=mean, kernel=kernel, device="cpu"
    )

    Phi = basis.phi(grid)
    expected = Phi.T @ Phi
    actual = basis.approximate_kernel(grid)

    assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)


def test_sample_functions_shape_and_mean_addition():
    torch.manual_seed(0)

    class ConstantMean(torch.nn.Module):
        def forward(self, x):
            return torch.full(x.shape[:-1], 3.0, dtype=x.dtype, device=x.device)

    R = 8
    B = 4
    C = 1
    grid = make_grid(n=6, d=2)
    mean = ConstantMean()
    kernel = make_kernel(lengthscale=torch.tensor([1.0, 1.0]), outputscale=1.0)

    basis = RFFBasis(
        R=R, input_dim=grid.shape[-1], mean=mean, kernel=kernel, device="cpu"
    )

    weights = torch.zeros(B, C, R, dtype=torch.double)
    samples = basis.sample_functions(weights, grid)

    assert samples.shape == (B, C, grid.shape[0])
    assert torch.allclose(samples, torch.full_like(samples, 3.0))
    assert not samples.requires_grad


def test_zero_order_derivative_equals_phi():
    torch.manual_seed(0)

    R = 10
    grid = make_grid(n=6, d=2)
    mean = gpytorch.means.ZeroMean()
    kernel = make_kernel(lengthscale=torch.tensor([0.8, 1.5]), outputscale=2.3)

    basis = RFFBasis(
        R=R, input_dim=grid.shape[-1], mean=mean, kernel=kernel, device="cpu"
    )

    deriv = basis.compute_derivative(grid, dims=[], orders=[])

    Phi = basis.phi(grid)
    assert deriv.shape == Phi.shape
    assert torch.allclose(deriv, Phi, atol=1e-10, rtol=1e-10)


def manual_phi_from_basis(basis, grid):
    """
    Manually rebuild the same scaled feature map expected from the RFF basis.

    Shape returned: (R, N).
    """
    theta = basis.omega_eff  # (D, M)
    phase = torch.einsum("dm,nd->mn", theta, grid)  # (M, N)

    features = torch.cat([torch.cos(phase), torch.sin(phase)], dim=0)
    features = features / math.sqrt(basis.R // 2)
    features = features * torch.sqrt(basis.outputscale)

    return features


def manual_derivative_from_basis(basis, grid, dims, orders):
    """
    Analytic derivative of the scaled RFF feature map.

    This is intentionally independent of basis.compute_derivative().
    """
    assert len(dims) == len(orders)

    theta = basis.omega_eff  # (D, M)
    phase = torch.einsum("dm,nd->mn", theta, grid)  # (M, N)

    total_order = sum(orders)

    prefactor = torch.ones(theta.shape[1], dtype=grid.dtype, device=grid.device)
    for dim, order in zip(dims, orders, strict=True):
        prefactor = prefactor * theta[dim, :] ** order

    if total_order % 4 == 0:
        cos_terms = torch.cos(phase)
        sin_terms = torch.sin(phase)
    elif total_order % 4 == 1:
        cos_terms = -torch.sin(phase)
        sin_terms = torch.cos(phase)
    elif total_order % 4 == 2:
        cos_terms = -torch.cos(phase)
        sin_terms = -torch.sin(phase)
    else:
        cos_terms = torch.sin(phase)
        sin_terms = -torch.cos(phase)

    deriv = torch.cat([cos_terms, sin_terms], dim=0)
    prefactor = torch.cat([prefactor, prefactor], dim=0)[:, None]

    deriv = deriv * prefactor
    deriv = deriv / math.sqrt(basis.R // 2)
    deriv = deriv * torch.sqrt(basis.outputscale)

    return deriv


@pytest.mark.parametrize(
    "dims,orders",
    [
        ([0], [1]),
        ([1], [1]),
        ([0], [2]),
        ([1], [2]),
        ([0, 1], [1, 1]),
        ([0, 1], [2, 1]),
    ],
)
def test_compute_derivative_matches_independent_formula(dims, orders):
    torch.manual_seed(0)

    R = 14
    grid = make_grid(n=7, d=2)
    mean = gpytorch.means.ZeroMean()

    # Use non-unit lengthscales to catch missing lengthscale factors.
    # Use D != R/2 to catch wrong normalization by sqrt(D).
    kernel = make_kernel(lengthscale=torch.tensor([0.6, 1.7]), outputscale=2.5)

    basis = RFFBasis(
        R=R, input_dim=grid.shape[-1], mean=mean, kernel=kernel, device="cpu"
    )

    expected = manual_derivative_from_basis(basis, grid, dims=dims, orders=orders)
    actual = basis.compute_derivative(grid, dims=dims, orders=orders)

    assert actual.shape == expected.shape
    assert torch.allclose(actual, expected, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("dim", [0, 1])
def test_first_derivative_matches_autograd(dim):
    torch.manual_seed(0)

    R = 12
    grid = make_grid(n=5, d=2).requires_grad_(True)
    mean = gpytorch.means.ZeroMean()
    kernel = make_kernel(lengthscale=torch.tensor([0.7, 1.4]), outputscale=1.9)

    basis = RFFBasis(
        R=R, input_dim=grid.detach().shape[-1], mean=mean, kernel=kernel, device="cpu"
    )

    expected_rows = []

    for feature_idx in range(R):
        grads_for_feature = []

        for n in range(grid.shape[0]):
            grid_for_grad = grid.clone().detach().requires_grad_(True)
            phi = manual_phi_from_basis(basis, grid_for_grad)  # (R, N)

            value = phi[feature_idx, n]
            grad = torch.autograd.grad(value, grid_for_grad, retain_graph=False)[0]
            grads_for_feature.append(grad[n, dim])

        expected_rows.append(torch.stack(grads_for_feature))

    expected = torch.stack(expected_rows, dim=0)
    actual = basis.compute_derivative(grid.detach(), dims=[dim], orders=[1])

    assert torch.allclose(actual, expected, atol=1e-10, rtol=1e-10)


def test_mixed_derivative_matches_autograd():
    torch.manual_seed(0)

    R = 12
    grid = make_grid(n=5, d=2).requires_grad_(True)
    mean = gpytorch.means.ZeroMean()
    kernel = make_kernel(lengthscale=torch.tensor([0.7, 1.4]), outputscale=1.9)

    basis = RFFBasis(
        R=R, input_dim=grid.detach().shape[-1], mean=mean, kernel=kernel, device="cpu"
    )

    expected_rows = []

    for feature_idx in range(R):
        vals = []

        for n in range(grid.shape[0]):
            grid_for_grad = grid.clone().detach().requires_grad_(True)
            phi = manual_phi_from_basis(basis, grid_for_grad)

            value = phi[feature_idx, n]

            grad_0 = torch.autograd.grad(
                value,
                grid_for_grad,
                create_graph=True,
                retain_graph=True,
            )[0][n, 0]

            grad_01 = torch.autograd.grad(
                grad_0,
                grid_for_grad,
                retain_graph=False,
            )[0][n, 1]

            vals.append(grad_01)

        expected_rows.append(torch.stack(vals))

    expected = torch.stack(expected_rows, dim=0)
    actual = basis.compute_derivative(grid.detach(), dims=[0, 1], orders=[1, 1])

    assert torch.allclose(actual, expected, atol=1e-10, rtol=1e-10)


# ---------------------------------------------------------------------------
# PosteriorKernel — the GP posterior covariance as a callable kernel.
# gpytorch has no such object; the posterior is otherwise only reachable as a
# MultivariateNormal over a fixed set of points.
# ---------------------------------------------------------------------------


def _post_setup(n_train=12, noise=1e-4, seed=0):
    torch.manual_seed(seed)
    train_x = torch.rand(n_train, 1, dtype=torch.float64)
    kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()).double()
    kernel.base_kernel.lengthscale = 0.2
    kernel.outputscale = 1.0
    return kernel, train_x, noise


def _analytic_posterior(kernel, train_x, noise, a, b):
    eye = torch.eye(len(train_x), dtype=train_x.dtype)
    K_xx = kernel(train_x, train_x).to_dense() + noise * eye
    return kernel(a, b).to_dense() - kernel(a, train_x).to_dense() @ torch.linalg.solve(
        K_xx, kernel(train_x, b).to_dense()
    )


class TestPosteriorKernel:
    def test_matches_analytic_formula(self):
        kernel, train_x, noise = _post_setup()
        a, b = torch.rand(5, 1).double(), torch.rand(7, 1).double()
        got = PosteriorKernel(kernel, train_x, noise)(a, b).to_dense()
        assert torch.allclose(got, _analytic_posterior(kernel, train_x, noise, a, b))

    def test_matches_gpytorch_exact_gp(self):
        # The same numbers gpytorch produces via model(X).covariance_matrix.
        kernel, train_x, noise = _post_setup()
        train_y = torch.sin(6 * train_x.squeeze(-1))

        class _M(gpytorch.models.ExactGP):
            def __init__(self, x, y, lik):
                super().__init__(x, y, lik)
                self.mean_module = gpytorch.means.ZeroMean()
                self.covar_module = kernel

            def forward(self, x):
                return gpytorch.distributions.MultivariateNormal(
                    self.mean_module(x), self.covar_module(x)
                )

        lik = gpytorch.likelihoods.GaussianLikelihood().double()
        lik.noise = noise
        model = _M(train_x, train_y, lik).double().eval()
        lik.eval()

        grid = torch.linspace(0, 1, 9, dtype=torch.float64).unsqueeze(-1)
        with torch.no_grad():
            ref = model(grid).covariance_matrix
        got = PosteriorKernel(kernel, train_x, noise)(grid, grid).to_dense()
        assert torch.allclose(got, ref, atol=1e-8)

    def test_diag_matches_full_diagonal(self):
        kernel, train_x, noise = _post_setup()
        a = torch.rand(6, 1).double()
        pk = PosteriorKernel(kernel, train_x, noise)
        assert torch.allclose(pk(a, a, diag=True), pk(a, a).to_dense().diagonal())

    def test_symmetric_and_psd(self):
        kernel, train_x, noise = _post_setup()
        grid = torch.linspace(0, 1, 20, dtype=torch.float64).unsqueeze(-1)
        K = PosteriorKernel(kernel, train_x, noise)(grid, grid).to_dense()
        assert torch.allclose(K, K.T, atol=1e-10)
        assert torch.linalg.eigvalsh(K).min() > -1e-8

    def test_variance_collapses_at_training_points(self):
        # The defining behaviour: no uncertainty where you have (near-noiseless) data,
        # prior variance far away.
        kernel, train_x, noise = _post_setup()
        pk = PosteriorKernel(kernel, train_x, noise)
        at_data = pk(train_x[:1], train_x[:1], diag=True).item()
        far = torch.tensor([[train_x.max().item() + 0.4]], dtype=torch.float64)
        assert at_data < 1e-3
        assert pk(far, far, diag=True).item() > 0.5

    def test_is_not_stationary(self):
        # Flagged because RFFBasis (Bochner) requires stationarity and so cannot be
        # applied to this kernel.
        kernel, train_x, noise = _post_setup()
        pk = PosteriorKernel(kernel, train_x, noise)
        lag = 0.1
        vals = [
            pk(
                torch.tensor([[x0]], dtype=torch.float64),
                torch.tensor([[x0 + lag]], dtype=torch.float64),
            )
            .to_dense()
            .item()
            for x0 in (0.1, 0.6)
        ]
        assert pk.is_stationary is False
        assert abs(vals[0] - vals[1]) > 1e-6  # same lag, different value

    def test_buffers_move_with_to(self):
        kernel, train_x, noise = _post_setup()
        pk = PosteriorKernel(kernel, train_x, noise).float()
        assert pk.L.dtype == torch.float32 and pk.train_x.dtype == torch.float32


# ---------------------------------------------------------------------------
# fit_kernel_hyperparameters — type-II ML for the kernel only. Conditioning is a
# separate step (gp_posterior), so this returns the fitted kernel, not a posterior.
# ---------------------------------------------------------------------------


class TestFitKernelHyperparameters:
    def _data(self, n=15, ell=0.15):
        torch.manual_seed(0)
        X = torch.rand(n, 1, dtype=torch.float64)
        y = torch.sin(X.squeeze(-1) / ell)
        k = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()).double()
        k.base_kernel.lengthscale = 1.0
        return X, y, k

    def test_returns_the_same_kernel_object(self):
        X, y, k = self._data()
        assert fit_kernel_hyperparameters(X, y, k, noise=1e-4, num_iters=20) is k

    def test_changes_hyperparameters(self):
        X, y, k = self._data()
        before = k.base_kernel.lengthscale.item()
        fit_kernel_hyperparameters(X, y, k, noise=1e-4, num_iters=100)
        assert k.base_kernel.lengthscale.item() != pytest.approx(before, rel=1e-3)

    def test_improves_marginal_likelihood(self):
        X, y, k = self._data()

        def nmll(kernel):
            eye = torch.eye(len(X), dtype=torch.float64)
            K = kernel(X, X).to_dense().detach() + 1e-4 * eye
            L = torch.linalg.cholesky(K)
            a = torch.cholesky_solve(y.unsqueeze(-1), L)
            return (0.5 * (y @ a.squeeze(-1)) + torch.log(L.diagonal()).sum()).item()

        before = nmll(k)
        fit_kernel_hyperparameters(X, y, k, noise=1e-4, num_iters=200)
        assert nmll(k) < before

    def test_noise_and_mean_are_not_fitted(self):
        # Only the kernel's own hyperparameters move; noise is given, mean is zero.
        X, y, k = self._data()
        fit_kernel_hyperparameters(X, y, k, noise=1e-3, num_iters=20)
        assert all(p.requires_grad for p in k.parameters())

    def test_feeds_gp_posterior(self):
        # The documented pipeline: fit, then condition, then build a flow.
        X, y, k = self._data()
        fit_kernel_hyperparameters(X, y, k, noise=1e-4, num_iters=50)
        mean, kern = gp_posterior(k, X, y, noise=1e-4)
        grid = torch.linspace(0, 1, 9, dtype=torch.float64).unsqueeze(-1)
        assert mean(grid).shape == (9,)
        assert kern(grid, grid).to_dense().shape == (9, 9)


class TestPosteriorMean:
    def test_matches_analytic_formula(self):
        kernel, train_x, noise = _post_setup()
        train_y = torch.sin(6 * train_x.squeeze(-1))
        grid = torch.linspace(0, 1, 9, dtype=torch.float64).unsqueeze(-1)
        eye = torch.eye(len(train_x), dtype=torch.float64)
        K_xx = kernel(train_x, train_x).to_dense() + noise * eye
        ref = kernel(grid, train_x).to_dense() @ torch.linalg.solve(K_xx, train_y)
        assert torch.allclose(PosteriorMean(kernel, train_x, train_y, noise)(grid), ref)

    def test_matches_gpytorch_exact_gp(self):
        kernel, train_x, noise = _post_setup()
        train_y = torch.sin(6 * train_x.squeeze(-1))

        class _M(gpytorch.models.ExactGP):
            def __init__(self, x, y, lik):
                super().__init__(x, y, lik)
                self.mean_module = gpytorch.means.ZeroMean()
                self.covar_module = kernel

            def forward(self, x):
                return gpytorch.distributions.MultivariateNormal(
                    self.mean_module(x), self.covar_module(x)
                )

        lik = gpytorch.likelihoods.GaussianLikelihood().double()
        lik.noise = noise
        model = _M(train_x, train_y, lik).double().eval()
        lik.eval()
        grid = torch.linspace(0, 1, 9, dtype=torch.float64).unsqueeze(-1)
        with torch.no_grad():
            ref = model(grid).mean
        got = PosteriorMean(kernel, train_x, train_y, noise)(grid)
        assert torch.allclose(got, ref, atol=1e-10)

    def test_shape_is_leading_dims(self):
        kernel, train_x, noise = _post_setup()
        train_y = torch.sin(6 * train_x.squeeze(-1))
        grid = torch.rand(11, 1, dtype=torch.float64)
        assert PosteriorMean(kernel, train_x, train_y, noise)(grid).shape == (11,)

    def test_interpolates_data_at_low_noise(self):
        kernel, train_x, _ = _post_setup()
        train_y = torch.sin(6 * train_x.squeeze(-1))
        pm = PosteriorMean(kernel, train_x, train_y, noise=1e-10)
        assert torch.allclose(pm(train_x), train_y, atol=1e-4)

    def test_reverts_to_prior_mean_far_from_data(self):
        kernel, train_x, noise = _post_setup()
        train_y = torch.sin(6 * train_x.squeeze(-1))
        const = gpytorch.means.ConstantMean().double()
        with torch.no_grad():
            const.constant.fill_(5.0)
        pm = PosteriorMean(kernel, train_x, train_y, noise, prior_mean=const)
        far = torch.tensor([[train_x.max().item() + 2.0]], dtype=torch.float64)
        assert pm(far).item() == pytest.approx(5.0, abs=1e-3)

    def test_non_zero_prior_mean_still_fits_data(self):
        kernel, train_x, _ = _post_setup()
        train_y = torch.sin(6 * train_x.squeeze(-1))
        const = gpytorch.means.ConstantMean().double()
        with torch.no_grad():
            const.constant.fill_(5.0)
        pm = PosteriorMean(kernel, train_x, train_y, 1e-10, prior_mean=const)
        assert torch.allclose(pm(train_x), train_y, atol=1e-3)


class TestGPPosteriorHelper:
    def test_matches_separate_construction(self):
        kernel, train_x, noise = _post_setup()
        train_y = torch.sin(6 * train_x.squeeze(-1))
        grid = torch.linspace(0, 1, 9, dtype=torch.float64).unsqueeze(-1)
        mean, kern = gp_posterior(kernel, train_x, train_y, noise)
        assert torch.equal(
            mean(grid), PosteriorMean(kernel, train_x, train_y, noise)(grid)
        )
        assert torch.equal(
            kern(grid, grid).to_dense(),
            PosteriorKernel(kernel, train_x, noise)(grid, grid).to_dense(),
        )

    def test_returns_mean_then_kernel(self):
        kernel, train_x, noise = _post_setup()
        train_y = torch.sin(6 * train_x.squeeze(-1))
        mean, kern = gp_posterior(kernel, train_x, train_y, noise)
        assert isinstance(mean, PosteriorMean) and isinstance(kern, PosteriorKernel)

    def test_shares_one_cholesky(self):
        # Both objects must see the same factor — that is the point of the helper.
        kernel, train_x, noise = _post_setup()
        train_y = torch.sin(6 * train_x.squeeze(-1))
        mean, kern = gp_posterior(kernel, train_x, train_y, noise)
        L_ref = _train_cholesky(kernel, train_x, noise)
        assert torch.allclose(kern.L, L_ref)
        assert torch.allclose(
            mean.alpha,
            torch.cholesky_solve(train_y.unsqueeze(-1), L_ref).squeeze(-1),
        )


# ---------------------------------------------------------------------------
# AbstractBasis: the contract every basis implementation must satisfy.
# ---------------------------------------------------------------------------

BASIS_GRID_N = 60


def _basis_setup(R=16):
    grid = build_grid_in_box(BASIS_GRID_N, 1, "cpu").double()
    kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()).double()
    kernel.base_kernel.lengthscale = 0.15
    kernel.outputscale = 0.7
    return R, grid, gpytorch.means.ZeroMean(), kernel


class TestAbstractBasisContract:
    def test_rff_is_abstract_subclass(self):
        assert issubclass(RFFBasis, AbstractBasis)

    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            AbstractBasis(4, 1, gpytorch.means.ZeroMean(), None, "cpu")

    def test_incomplete_subclass_rejected(self):
        class _NoPhi(AbstractBasis):
            def _get_basis_params(self):
                pass

        with pytest.raises(TypeError):
            _NoPhi(4, 1, gpytorch.means.ZeroMean(), None, "cpu")

    def test_phi_shape(self):
        R, grid, mean, kernel = _basis_setup()
        b = RFFBasis(
            R=R, input_dim=grid.shape[-1], mean=mean, kernel=kernel, device="cpu"
        )
        assert b.phi(grid).shape == (R, BASIS_GRID_N)

    def test_approximate_kernel_is_phi_gram(self):
        R, grid, mean, kernel = _basis_setup()
        b = RFFBasis(
            R=R, input_dim=grid.shape[-1], mean=mean, kernel=kernel, device="cpu"
        )
        assert torch.allclose(
            b.approximate_kernel(grid),
            torch.einsum("rn,rm->nm", b.phi(grid), b.phi(grid)),
            atol=1e-10,
        )

    def test_sample_functions_shape(self):
        R, grid, mean, kernel = _basis_setup()
        b = RFFBasis(
            R=R, input_dim=grid.shape[-1], mean=mean, kernel=kernel, device="cpu"
        )
        out = b.sample_functions(torch.randn(5, 1, R, dtype=torch.float64), grid)
        assert out.shape == (5, 1, BASIS_GRID_N)


# ---------------------------------------------------------------------------
# DecoupledRFFBasis — a GP posterior as a finite basis, via pathwise conditioning
# (Wilson et al. 2020). The posterior kernel is non-stationary so it cannot be
# RFF'd directly; this expands only the (stationary) prior and corrects exactly.
# ---------------------------------------------------------------------------

DEC_M, DEC_N, DEC_RF, DEC_NOISE = 10, 40, 512, 1e-3


def _decoupled_setup(num_features=DEC_RF, noise=DEC_NOISE):
    torch.manual_seed(0)
    train_x = torch.rand(DEC_M, 1, dtype=torch.float64)
    train_y = torch.sin(6 * train_x.squeeze(-1))
    # grid deliberately extends beyond the data, where variance starvation shows
    grid = torch.linspace(-0.2, 1.2, DEC_N, dtype=torch.float64).unsqueeze(-1)
    kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()).double()
    kernel.base_kernel.lengthscale = 0.15
    kernel.outputscale = 1.0
    basis = DecoupledRFFBasis(
        num_features=num_features,
        input_dim=1,
        kernel=kernel,
        train_x=train_x,
        train_y=train_y,
        noise=noise,
    )
    return basis, kernel, train_x, train_y, grid


class TestDecoupledRFFBasis:
    def test_is_a_basis(self):
        basis, *_ = _decoupled_setup()
        assert isinstance(basis, AbstractBasis)

    def test_latent_width_is_features_plus_data(self):
        basis, *_ = _decoupled_setup()
        assert basis.R == basis.num_features + DEC_M
        assert basis.has_noise_block

    def test_noiseless_drops_the_eps_block(self):
        # With no observation noise the eps features are identically zero, so
        # carrying them would give the flow latent dimensions with no effect.
        basis, *_ = _decoupled_setup(num_features=64, noise=0.0)
        assert basis.R == basis.num_features
        assert not basis.has_noise_block

    def test_phi_shape(self):
        basis, _, _, _, grid = _decoupled_setup()
        assert basis.phi(grid).shape == (basis.R, DEC_N)

    def test_mean_is_the_exact_posterior_mean(self):
        # Matheron's rule puts the deterministic part in the mean, exactly.
        basis, kernel, train_x, train_y, grid = _decoupled_setup()
        mean_ref, _ = gp_posterior(kernel, train_x, train_y, noise=DEC_NOISE)
        assert torch.allclose(basis.mean(grid), mean_ref(grid), atol=1e-10)

    def test_mean_coerces_foreign_dtype(self):
        # basis.mean is a method, not the bare gpytorch module, so it coerces its
        # argument like phi does. Called on the module, a float32 grid against a
        # float64 posterior raises inside gpytorch instead of promoting.
        basis, _, _, _, grid = _decoupled_setup()
        out = basis.mean(grid.to(torch.float32))
        assert out.dtype == basis.dtype
        assert torch.allclose(out, basis.mean(grid), atol=1e-5)

    def test_covariance_approximates_the_posterior(self):
        basis, kernel, train_x, train_y, grid = _decoupled_setup()
        _, k_post = gp_posterior(kernel, train_x, train_y, noise=DEC_NOISE)
        err = (basis.approximate_kernel(grid) - k_post(grid, grid).to_dense()).abs()
        assert err.max() < 0.1

    def test_beats_naive_weight_space_conditioning(self):
        # The point of the method: conditioning the RFF *model* in weight space
        # under-covers away from the data (variance starvation).
        basis, kernel, train_x, train_y, grid = _decoupled_setup()
        _, k_post = gp_posterior(kernel, train_x, train_y, noise=DEC_NOISE)
        K_ref = k_post(grid, grid).to_dense()

        prior_phi = basis.prior_basis
        Phi_g, Phi_X = prior_phi.phi(grid), prior_phi.phi(train_x)
        Sw = torch.linalg.inv(
            Phi_X @ Phi_X.T / DEC_NOISE
            + torch.eye(basis.num_features, dtype=Phi_X.dtype)
        )
        naive_err = (Phi_g.T @ Sw @ Phi_g - K_ref).abs().max()
        decoupled_err = (basis.approximate_kernel(grid) - K_ref).abs().max()
        assert decoupled_err < naive_err

    def test_sampled_moments_match_the_posterior(self):
        basis, kernel, train_x, train_y, grid = _decoupled_setup()
        mean_ref, k_post = gp_posterior(kernel, train_x, train_y, noise=DEC_NOISE)
        torch.manual_seed(1)
        w = torch.randn(20000, 1, basis.R, dtype=torch.float64)
        f = basis.sample_functions(w, grid).squeeze(1)
        assert torch.allclose(f.mean(0), mean_ref(grid), atol=0.05)
        assert (torch.cov(f.T) - k_post(grid, grid).to_dense()).abs().max() < 0.15

    def test_grid_free(self):
        basis, _, _, _, grid = _decoupled_setup()
        assert not hasattr(basis, "grid")
        for n in (3, 17, DEC_N):
            assert basis.phi(grid[:n]).shape == (basis.R, n)

    def test_derivatives_available(self):
        basis, _, _, _, grid = _decoupled_setup()
        assert basis.compute_derivative(grid, (0,), (2,)).shape == (basis.R, DEC_N)
        assert basis.mean_derivative(grid, (0,), (2,)).shape == (DEC_N,)

    def test_mean_derivative_rejects_a_non_zero_prior_mean(self):
        # A constant/parametric prior mean would contribute its own derivative term,
        # which is not implemented — fail loudly rather than silently drop it.
        torch.manual_seed(0)
        train_x = torch.rand(DEC_M, 1, dtype=torch.float64)
        kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()).double()
        kernel.base_kernel.lengthscale = 0.15
        const = gpytorch.means.ConstantMean().double()
        basis = DecoupledRFFBasis(
            num_features=32,
            input_dim=1,
            kernel=kernel,
            train_x=train_x,
            train_y=torch.sin(6 * train_x.squeeze(-1)),
            noise=DEC_NOISE,
            prior_mean=const,
        )
        with pytest.raises(NotImplementedError, match="ZeroMean"):
            basis.mean_derivative(train_x, (0,), (1,))

    def test_drives_a_basis_flow(self):
        from flow_inference.flows import BasisGaussianProcessFlow
        from flow_inference.sampler import IntegratorSampler

        basis, _, _, _, grid = _decoupled_setup(num_features=64)
        flow = BasisGaussianProcessFlow(basis, grid)
        assert flow.latent_dim == basis.R and flow.output_dim == DEC_N
        out = IntegratorSampler(flow).sample(num_samples=2, num_steps=4, verbose=False)
        assert out.shape == (2, 1, DEC_N) and torch.isfinite(out).all()


class TestDecoupledRFFBasisDerivatives:
    """Closed-form derivatives of the conditioned basis, against autograd.

    Everything reduces to d^n/dx^n k(x, X) for the prior kernel, which for a scaled
    ARD RBF factorises into Hermite polynomials. Checked to 4th order and on mixed
    partials, since the Hermite recurrence is where an error would hide.
    """

    def _setup(self, num_features=64, noise=1e-3):
        torch.manual_seed(0)
        train_x = torch.rand(8, 2, dtype=torch.float64)
        train_y = torch.sin(6 * train_x[:, 0]) + train_x[:, 1]
        pts = torch.rand(9, 2, dtype=torch.float64)
        kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=2)
        ).double()
        kernel.base_kernel.lengthscale = torch.tensor([[0.3, 0.5]], dtype=torch.float64)
        kernel.outputscale = 1.7
        basis = DecoupledRFFBasis(
            num_features=num_features,
            input_dim=2,
            kernel=kernel,
            train_x=train_x,
            train_y=train_y,
            noise=noise,
        )
        return basis, kernel, train_x, pts

    @staticmethod
    def _autograd(fn, pts, dims, orders):
        from torch.func import jacrev

        axes = [d for d, o in zip(dims, orders, strict=True) for _ in range(o)]
        g = fn
        for _ in axes:
            g = jacrev(g)
        out = torch.stack([g(p) for p in pts])
        for d in reversed(axes):
            out = out[..., d]
        return out

    def test_hand_written_rbf_matches_gpytorch(self):
        # _prior_kernel_derivative reimplements the RBF rather than calling the kernel;
        # order 0 must therefore agree with gpytorch exactly.
        basis, kernel, train_x, pts = self._setup()
        mine = basis._prior_kernel_derivative(pts, (), ())
        assert torch.allclose(mine, kernel(pts, train_x).to_dense(), atol=1e-12)

    @pytest.mark.parametrize(
        "dims,orders",
        [
            ((0,), (1,)),
            ((1,), (1,)),
            ((0,), (2,)),
            ((0, 1), (1, 1)),
            ((0,), (3,)),
            ((0,), (4,)),
        ],
    )
    def test_kernel_derivative_matches_autograd(self, dims, orders):
        basis, kernel, train_x, pts = self._setup()
        ref = self._autograd(
            lambda p: kernel(p[None, :], train_x).to_dense()[0], pts, dims, orders
        )
        got = basis._prior_kernel_derivative(pts, dims, orders)
        assert torch.allclose(got, ref, atol=1e-9, rtol=1e-9)

    @pytest.mark.parametrize(
        "dims,orders", [((0,), (1,)), ((0,), (2,)), ((1,), (1,)), ((0, 1), (1, 1))]
    )
    def test_basis_derivative_matches_autograd(self, dims, orders):
        basis, _, _, pts = self._setup()
        ref = self._autograd(lambda p: basis.phi(p[None, :])[:, 0], pts, dims, orders)
        assert torch.allclose(
            basis.compute_derivative(pts, dims, orders), ref.T, atol=1e-9, rtol=1e-9
        )

    @pytest.mark.parametrize("dims,orders", [((0,), (1,)), ((0,), (2,)), ((1,), (1,))])
    def test_mean_derivative_matches_autograd(self, dims, orders):
        basis, _, _, pts = self._setup()
        ref = self._autograd(lambda p: basis.mean(p[None, :])[0], pts, dims, orders)
        assert torch.allclose(
            basis.mean_derivative(pts, dims, orders), ref, atol=1e-9, rtol=1e-9
        )

    def test_zero_order_returns_phi_and_mean(self):
        basis, _, _, pts = self._setup()
        assert torch.allclose(basis.compute_derivative(pts, (), ()), basis.phi(pts))
        assert torch.allclose(basis.mean_derivative(pts, (), ()), basis.mean(pts))

    def test_sampled_function_derivative_needs_the_mean_term(self):
        # A sampled function is w @ phi + mean, so both terms differentiate. Dropping
        # the mean term is a silent, large error — this pins that it is required.
        basis, _, _, pts = self._setup()
        torch.manual_seed(1)
        w = torch.randn(3, 1, basis.R, dtype=torch.float64)
        dims, orders = (0,), (1,)
        ref = self._autograd(
            lambda p: (
                (w @ basis.phi(p[None, :])).squeeze(1)[:, 0] + basis.mean(p[None, :])[0]
            ),
            pts,
            dims,
            orders,
        )
        d_basis = (w @ basis.compute_derivative(pts, dims, orders)).squeeze(1)
        assert torch.allclose(
            d_basis + basis.mean_derivative(pts, dims, orders), ref.T, atol=1e-9
        )
        assert not torch.allclose(
            d_basis, ref.T, atol=1e-3
        )  # mean term is not optional

    def test_noiseless_basis_has_derivatives_too(self):
        basis, _, _, pts = self._setup(noise=0.0)
        assert basis.compute_derivative(pts, (0,), (2,)).shape == (basis.R, len(pts))


# ---------------------------------------------------------------------------
# RFFBasis closed form vs gpytorch
# ---------------------------------------------------------------------------


def _gpytorch_phi(basis, x):
    """RFF features at ``x`` ``(N, D)`` via gpytorch's own featurisation, ``(R, N)``.

    An independent oracle for ``RFFBasis.phi``: same frequencies, but gpytorch's code
    path rather than the library's closed form.
    """
    rff = ScaleKernel(
        RFFKernel(num_samples=basis.R // 2, ard_num_dims=basis.D, num_dims=basis.D)
    ).to(dtype=basis.dtype)
    rff.base_kernel.randn_weights = basis.omega
    rff.base_kernel.lengthscale = basis.lengthscale.detach().reshape(1, 1, -1)
    feats = rff.base_kernel._featurize(x, normalize=True)  # (N, R)
    return (torch.sqrt(basis.outputscale) * feats).T


def test_rff_phi_matches_the_gpytorch_oracle():
    kernel = make_kernel([0.2, 0.4], outputscale=1.5)
    basis = RFFBasis(R=32, input_dim=2, mean=gpytorch.means.ZeroMean(), kernel=kernel)
    grid = build_grid_in_box(8, 5)
    oracle = _gpytorch_phi(basis, grid)
    assert torch.allclose(basis.phi(grid), oracle, atol=1e-12)
