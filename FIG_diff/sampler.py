import time
import tqdm
import torch
import numpy as np
import torch.nn as nn
import torchvision

def get_sampler(**kwargs):
    latent = kwargs['latent']
    kwargs.pop('latent')
    if latent:
        raise NotImplementedError
    name = kwargs['name']
    kwargs.pop('name')
    if name == 'daps':
        return DAPS(**kwargs)
    elif name == 'fig':
        return FIG(**kwargs)
    elif name == 'dps':
        return DPS(**kwargs)
    else:
        raise NotImplementedError  


class Scheduler(nn.Module):
    """
        Scheduler for diffusion sigma(t) and discretization step size Delta t
    """

    def __init__(self, num_steps=10, sigma_max=100, sigma_min=0.01, sigma_final=None, schedule='linear',
                 timestep='poly-7'):
        """
            Initializes the scheduler with the given parameters.

            Parameters:
                num_steps (int): Number of steps in the schedule.
                sigma_max (float): Maximum value of sigma.
                sigma_min (float): Minimum value of sigma.
                sigma_final (float): Final value of sigma, defaults to sigma_min.
                schedule (str): Type of schedule for sigma ('linear' or 'sqrt').
                timestep (str): Type of timestep function ('log' or 'poly-n').
        """
        super().__init__()
        self.num_steps = num_steps
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.sigma_final = sigma_final
        if self.sigma_final is None:
            self.sigma_final = self.sigma_min
        self.schedule = schedule
        self.timestep = timestep

        steps = np.linspace(0, 1, num_steps)
        sigma_fn, sigma_derivative_fn, sigma_inv_fn = self.get_sigma_fn(self.schedule)
        time_step_fn = self.get_time_step_fn(self.timestep, self.sigma_max, self.sigma_min)

        time_steps = np.array([time_step_fn(s) for s in steps])
        time_steps = np.append(time_steps, sigma_inv_fn(self.sigma_final))
        sigma_steps = np.array([sigma_fn(t) for t in time_steps])

        # factor = 2\dot\sigma(t)\sigma(t)\Delta t
        factor_steps = np.array(
            [2 * sigma_fn(time_steps[i]) * sigma_derivative_fn(time_steps[i]) * (time_steps[i] - time_steps[i + 1]) for
             i in range(num_steps)])
        self.sigma_steps, self.time_steps, self.factor_steps = sigma_steps, time_steps, factor_steps
        self.factor_steps = [max(f, 0) for f in self.factor_steps]

    def get_sigma_fn(self, schedule):
        """
            Returns the sigma function, its derivative, and its inverse based on the given schedule.
        """
        if schedule == 'sqrt':
            sigma_fn = lambda t: np.sqrt(t)
            sigma_derivative_fn = lambda t: 1 / 2 / np.sqrt(t)
            sigma_inv_fn = lambda sigma: sigma ** 2

        elif schedule == 'linear':
            sigma_fn = lambda t: t
            sigma_derivative_fn = lambda t: 1
            sigma_inv_fn = lambda t: t
        else:
            raise NotImplementedError
        return sigma_fn, sigma_derivative_fn, sigma_inv_fn

    def get_time_step_fn(self, timestep, sigma_max, sigma_min):
        """
            Returns the time step function based on the given timestep type.
        """
        if timestep == 'log':
            get_time_step_fn = lambda r: sigma_max ** 2 * (sigma_min ** 2 / sigma_max ** 2) ** r
        elif timestep.startswith('poly'):
            p = int(timestep.split('-')[1])
            get_time_step_fn = lambda r: (sigma_max ** (1 / p) + r * (sigma_min ** (1 / p) - sigma_max ** (1 / p))) ** p
        else:
            raise NotImplementedError
        return get_time_step_fn


class DiffusionSampler(nn.Module):
    """
        Diffusion sampler for reverse SDE or PF-ODE
    """

    def __init__(self, scheduler, solver='euler'):
        """
            Initializes the diffusion sampler with the given scheduler and solver.

            Parameters:
                scheduler (Scheduler): Scheduler instance for managing sigma and timesteps.
                solver (str): Solver method ('euler').
        """
        super().__init__()
        self.scheduler = scheduler
        self.solver = solver

    def sample(self, model, x_start, SDE=False, record=False, verbose=False):
        """
            Samples from the diffusion process using the specified model.

            Parameters:
                model (DiffusionModel): Diffusion model supports 'score' and 'tweedie'
                x_start (torch.Tensor): Initial state.
                SDE (bool): Whether to use Stochastic Differential Equations.
                record (bool): Whether to record the trajectory.
                verbose (bool): Whether to display progress bar.

            Returns:
                torch.Tensor: The final sampled state.
        """
        if self.solver == 'euler':
            return self._euler(model, x_start, SDE, record, verbose)
        else:
            raise NotImplementedError

    def _euler(self, model, x_start, SDE=False, record=False, verbose=False):
        """
            Euler's method for sampling from the diffusion process.
        """
        if record:
            self.trajectory = Trajectory()
        pbar = tqdm.trange(self.scheduler.num_steps) if verbose else range(self.scheduler.num_steps)

        x = x_start
        for step in pbar:
            sigma, factor = self.scheduler.sigma_steps[step], self.scheduler.factor_steps[step]
            score = model.score(x, sigma)
            if SDE:
                epsilon = torch.randn_like(x)
                x = x + factor * score + np.sqrt(factor) * epsilon
            else:
                x = x + factor * score * 0.5
            # record
            if record:
                if SDE:
                    self._record(x, score, sigma, factor, epsilon)
                else:
                    self._record(x, score, sigma, factor)
        return x

    def _record(self, x, score, sigma, factor, epsilon=None):
        """
            Records the intermediate states during sampling.
        """
        self.trajectory.add_tensor(f'xt', x)
        self.trajectory.add_tensor(f'score', score)
        self.trajectory.add_value(f'sigma', sigma)
        self.trajectory.add_value(f'factor', factor)
        if epsilon is not None:
            self.trajectory.add_tensor(f'epsilon', epsilon)

    def get_start(self, ref):
        """
            Generates a random initial state based on the reference tensor.

            Parameters:
                ref (torch.Tensor): Reference tensor for shape and device.

            Returns:
                torch.Tensor: Initial random state.
        """
        x_start = torch.randn_like(ref) * self.scheduler.sigma_max
        return x_start


class LatentDiffusionSampler(DiffusionSampler):
    """
        Latent Diffusion sampler for reverse SDE or PF-ODE
    """

    def __init__(self, scheduler, solver='euler'):
        """
            Initializes the latent diffusion sampler with the given scheduler and solver.

            Parameters:
                scheduler (Scheduler): Scheduler instance for managing sigma and timesteps.
                solver (str): Solver method ('euler').
        """
        super().__init__(scheduler, solver)

    def sample(self, model, z_start, SDE=False, record=False, verbose=False, return_latent=True):
        """
            Samples from the latent diffusion process using the specified model.

            Parameters:
                model (LatentDiffusionModel): Diffusion model supports 'score', 'tweedie', 'encode' and 'decode'
                z_start (torch.Tensor): Initial latent state.
                SDE (bool): Whether to use Stochastic Differential Equations.
                record (bool): Whether to record the trajectory.
                verbose (bool): Whether to display progress bar.
                return_latent (bool): Whether to return the latent state or decoded state.

            Returns:
                torch.Tensor: The final sampled state (latent or decoded).
        """
        if self.solver == 'euler':
            z0 = self._euler(model, z_start, SDE, record, verbose)
        else:
            raise NotImplementedError
        if return_latent:
            return z0
        else:
            x0 = model.decode(z0)
            return x0


class LangevinDynamics(nn.Module):
    """
        Langevin Dynamics sampling method.
    """

    def __init__(self, num_steps, lr, tau=0.01, lr_min_ratio=0.01):
        """
            Initializes the Langevin dynamics sampler with the given parameters.

            Parameters:
                num_steps (int): Number of steps in the sampling process.
                lr (float): Learning rate.
                tau (float): Noise parameter.
                lr_min_ratio (float): Minimum learning rate ratio.
        """
        super().__init__()
        self.num_steps = num_steps
        self.lr = lr
        self.tau = tau
        self.lr_min_ratio = lr_min_ratio

    def sample(self, x0hat, operator, measurement, sigma, ratio, record=False, verbose=False):
        """
            Samples using Langevin dynamics.

            Parameters:
                x0hat (torch.Tensor): Initial state.
                operator (Operator): Operator module.
                measurement (torch.Tensor): Measurement tensor.
                sigma (float): Current sigma value.
                ratio (float): Current step ratio.
                record (bool): Whether to record the trajectory.
                verbose (bool): Whether to display progress bar.

            Returns:
                torch.Tensor: The final sampled state.
        """
        if record:
            self.trajectory = Trajectory()
        pbar = tqdm.trange(self.num_steps) if verbose else range(self.num_steps)
        lr = self.get_lr(ratio)
        x = x0hat.clone().detach().requires_grad_(True)
        optimizer = torch.optim.SGD([x], lr)
        for _ in pbar:
            optimizer.zero_grad()
            loss = operator.error(x, measurement).sum() / (2 * self.tau ** 2)
            loss += ((x - x0hat.detach()) ** 2).sum() / (2 * sigma ** 2)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                epsilon = torch.randn_like(x)
                x.data = x.data + np.sqrt(2 * lr) * epsilon

            # early stopping with NaN
            if torch.isnan(x).any():
                return torch.zeros_like(x)

            # record
            if record:
                self._record(x, epsilon, loss)
        return x.detach()

    def _record(self, x, epsilon, loss):
        """
            Records the intermediate states during sampling.
        """
        self.trajectory.add_tensor(f'xi', x)
        self.trajectory.add_tensor(f'epsilon', epsilon)
        self.trajectory.add_value(f'loss', loss)

    def get_lr(self, ratio):
        """
            Computes the learning rate based on the given ratio.
        """
        p = 1
        multiplier = (1 ** (1 / p) + ratio * (self.lr_min_ratio ** (1 / p) - 1 ** (1 / p))) ** p
        return multiplier * self.lr


class DAPS(nn.Module):
    """
        Implementation of decoupled annealing posterior sampling.
    """

    def __init__(self, annealing_scheduler_config, diffusion_scheduler_config, lgvd_config):
        """
            Initializes the DAPS sampler with the given configurations.

            Parameters:
                annealing_scheduler_config (dict): Configuration for annealing scheduler.
                diffusion_scheduler_config (dict): Configuration for diffusion scheduler.
                lgvd_config (dict): Configuration for Langevin dynamics.
        """
        super().__init__()
        annealing_scheduler_config, diffusion_scheduler_config = self._check(annealing_scheduler_config,diffusion_scheduler_config)
        self.annealing_scheduler = Scheduler(**annealing_scheduler_config)
        self.diffusion_scheduler_config = diffusion_scheduler_config
        self.lgvd = LangevinDynamics(**lgvd_config)

    def sample(self, model, x_start, operator, measurement, evaluator=None, record=False, verbose=False, **kwargs):
        """
            Samples using the DAPS method.

            Parameters:
                model (nn.Module): (Latent) Diffusion model
                x_start (torch.Tensor): Initial state.
                operator (nn.Module): Operator module.
                measurement (torch.Tensor): Measurement tensor.
                evaluator (Evaluator): Evaluation function.
                record (bool): Whether to record the trajectory.
                verbose (bool): Whether to display progress bar.
                **kwargs:
                    gt (torch.Tensor): reference ground truth data, only for evaluation

            Returns:
                torch.Tensor: The final sampled state.
        """
        if record:
            self.trajectory = Trajectory()
        pbar = tqdm.trange(self.annealing_scheduler.num_steps) if verbose else range(self.annealing_scheduler.num_steps)
        xt = x_start
        ode_time = 0
        ld_time = 0
        for step in pbar: # 200 steps
            sigma = self.annealing_scheduler.sigma_steps[step]
            # 1. reverse diffusion
            diffusion_scheduler = Scheduler(**self.diffusion_scheduler_config, sigma_max=sigma)
            sampler = DiffusionSampler(diffusion_scheduler)
            start_time = time.time()
            x0hat = sampler.sample(model, xt, SDE=False, verbose=False) # 5 steps
            end_time = time.time()
            ode_time += end_time-start_time

            # 2. langevin dynamics
            start_time = time.time()
            x0y = self.lgvd.sample(x0hat, operator, measurement, sigma, step / self.annealing_scheduler.num_steps) # 100 steps
            end_time = time.time()
            ld_time += end_time-start_time

            # 3. forward diffusion
            xt = x0y + torch.randn_like(x0y) * self.annealing_scheduler.sigma_steps[step + 1]

            # 4. evaluation
            x0hat_results = x0y_results = {}
            if evaluator and 'gt' in kwargs:
                with torch.no_grad():
                    gt = kwargs['gt']
                    x0hat_results = evaluator(gt, measurement, x0hat)
                    x0y_results = evaluator(gt, measurement, x0y)

                # record
                if verbose:
                    main_eval_fn_name = evaluator.main_eval_fn_name
                    pbar.set_postfix({
                        'x0hat' + '_' + main_eval_fn_name: f"{x0hat_results[main_eval_fn_name].item():.2f}",
                        'x0y' + '_' + main_eval_fn_name: f"{x0y_results[main_eval_fn_name].item():.2f}",
                    })
            if record:
                self._record(xt, x0y, x0hat, sigma, x0hat_results, x0y_results)
        
        print(f"ODE time {ode_time/self.annealing_scheduler.num_steps}")
        print(f"LD time {ld_time/self.annealing_scheduler.num_steps}")
        
        return xt

    def _record(self, xt, x0y, x0hat, sigma, x0hat_results, x0y_results):
        """
            Records the intermediate states during sampling.
        """
        self.trajectory.add_tensor(f'xt', xt)
        self.trajectory.add_tensor(f'x0y', x0y)
        self.trajectory.add_tensor(f'x0hat', x0hat)
        self.trajectory.add_value(f'sigma', sigma)
        for name in x0hat_results.keys():
            self.trajectory.add_value(f'x0hat_{name}', x0hat_results[name])
        for name in x0y_results.keys():
            self.trajectory.add_value(f'x0y_{name}', x0y_results[name])

    def _check(self, annealing_scheduler_config, diffusion_scheduler_config):
        """
            Checks and updates the configurations for the schedulers.
        """
        # sigma_max of diffusion scheduler change each step
        if 'sigma_max' in diffusion_scheduler_config:
            diffusion_scheduler_config.pop('sigma_max')

        # sigma final of annealing scheduler should always be 0
        annealing_scheduler_config['sigma_final'] = 0
        return annealing_scheduler_config, diffusion_scheduler_config

    def get_start(self, ref):
        """
            Generates a random initial state based on the reference tensor.

            Parameters:
                ref (torch.Tensor): Reference tensor for shape and device.

            Returns:
                torch.Tensor: Initial random state.
        """
        x_start = torch.randn_like(ref) * self.annealing_scheduler.sigma_max
        return x_start


class Trajectory(nn.Module):
    """
        Class for recording and storing trajectory data.
    """

    def __init__(self):
        super().__init__()
        self.tensor_data = {}
        self.value_data = {}
        self._compile = False

    def add_tensor(self, name, images):
        """
            Adds image data to the trajectory.

            Parameters:
                name (str): Name of the image data.
                images (torch.Tensor): Image tensor to add.
        """
        if name not in self.tensor_data:
            self.tensor_data[name] = []
        self.tensor_data[name].append(images.detach().cpu())

    def add_value(self, name, values):
        """
            Adds value data to the trajectory.

            Parameters:
                name (str): Name of the value data.
                values (any): Value to add.
        """
        if name not in self.value_data:
            self.value_data[name] = []
        self.value_data[name].append(values)

    def compile(self):
        """
            Compiles the recorded data into tensors.

            Returns:
                Trajectory: The compiled trajectory object.
        """
        if not self._compile:
            self._compile = True
            for name in self.tensor_data.keys():
                self.tensor_data[name] = torch.stack(self.tensor_data[name], dim=0)
            for name in self.value_data.keys():
                self.value_data[name] = torch.tensor(self.value_data[name])
        return self

    @classmethod
    def merge(cls, trajs):
        """
            Merge a list of compiled trajectories from different batches

            Returns:
                Trajectory: The merged and compiled trajectory object.
        """
        merged_traj = cls()
        for name in trajs[0].tensor_data.keys():
            merged_traj.tensor_data[name] = torch.cat([traj.tensor_data[name] for traj in trajs], dim=1)
        for name in trajs[0].value_data.keys():
            merged_traj.value_data[name] = trajs[0].value_data[name]
        return merged_traj


class DPSSampler(nn.Module):
    """
        FIG sampler for reverse SDE or PF-ODE
    """

    def __init__(self, param, scheduler, solver='euler'):
        """
            Initializes the diffusion sampler with the given scheduler and solver.

            Parameters:
                scheduler (Scheduler): Scheduler instance for managing sigma and timesteps.
                solver (str): Solver method ('euler').
        """
        super().__init__()
        self.param = param
        self.scheduler = scheduler
        self.solver = solver

    def sample(self, model, x_start, operator, measurement, SDE=False, record=False, verbose=False, **kwargs):
        """
            Samples from the diffusion process using the specified model.

            Parameters:
                model (DiffusionModel): Diffusion model supports 'score' and 'tweedie'
                x_start (torch.Tensor): Initial state.
                operator (Operator): Operator module.
                measurement (torch.Tensor): Measurement tensor.
                K (int): correction steps.
                lr (float): gradient descent learning rate 
                SDE (bool): Whether to use Stochastic Differential Equations.
                record (bool): Whether to record the trajectory.
                verbose (bool): Whether to display progress bar.

            Returns:
                torch.Tensor: The final sampled state.
        """
        if self.solver == 'euler':
            return self._euler(model, x_start, operator, measurement, SDE, record, verbose)
        else:
            raise NotImplementedError

    def _euler(self, model, x_start, operator, measurement, SDE=False, record=False, verbose=False):
        """
            Euler's method for sampling from the diffusion process.
        """
        if record:
            self.trajectory = Trajectory()
        pbar = tqdm.tqdm(range(self.scheduler.num_steps))# if verbose else range(self.scheduler.num_steps)
        
        x = x_start
        y = measurement
        lr = self.param.get('lr', 0.5)
        for step in pbar:
            sigma, factor = self.scheduler.sigma_steps[step], self.scheduler.factor_steps[step]
            x.requires_grad_(True)
            score = model.score(x, sigma)
            x0hat = x + score * sigma ** 2
            if step > 0:
                norm = operator.error(x0hat, y).sum()
                norm_grad = torch.autograd.grad(outputs=norm, inputs=x)[0]
                x = x - norm_grad  * lr
            x = x.detach()
            score = score.detach()

            if SDE:
                epsilon = torch.randn_like(x)
                x = x + factor * score + np.sqrt(factor) * epsilon
            else:
                x = x + factor * score * 0.5
            # record
            if record:
                if SDE:
                    self._record(x, score, sigma, factor, epsilon)
                else:
                    self._record(x, score, sigma, factor)
            
            # print("x:", x.max(), x.min())
            # print("y:", y.max(), y.min())
            # print("factor", factor)

            # pbar.set_postfix({'max': x.max().item(),'min': x.min().item()}, refresh=False)
            pbar.set_postfix({'dist': operator.error(x0hat,y).sum()}, refresh=False)
            # if (step + 1) % 10 == 0:
                # torchvision.utils.save_image(y.clamp(-1,1) / 2 + 0.5, f"./tmp/y_{step}.png")

        # torchvision.utils.save_image(x.clamp(-1,1) / 2 + 0.5, f"./tmp/x_{step}.png")

        return x

    def _record(self, x, score, sigma, factor, epsilon=None):
        """
            Records the intermediate states during sampling.
        """
        self.trajectory.add_tensor(f'xt', x)
        self.trajectory.add_tensor(f'score', score)
        self.trajectory.add_value(f'sigma', sigma)
        self.trajectory.add_value(f'factor', factor)
        if epsilon is not None:
            self.trajectory.add_tensor(f'epsilon', epsilon)

    def get_start(self, ref):
        """
            Generates a random initial state based on the reference tensor.

            Parameters:
                ref (torch.Tensor): Reference tensor for shape and device.

            Returns:
                torch.Tensor: Initial random state.
        """
        x_start = torch.randn_like(ref) * self.scheduler.sigma_max
        return x_start


class DPS(nn.Module):
    """
        Implementation of decoupled annealing posterior sampling.
    """

    def __init__(self, algo_param, diffusion_scheduler_config, **kwargs):
        """
            Initializes the DAPS sampler with the given configurations.

            Parameters:
                annealing_scheduler_config (dict): Configuration for annealing scheduler.
                diffusion_scheduler_config (dict): Configuration for diffusion scheduler.
                lgvd_config (dict): Configuration for Langevin dynamics.
        """
        super().__init__()
        self.algo_param = algo_param
        self.diffusion_scheduler = Scheduler(**diffusion_scheduler_config)

    def sample(self, model, x_start, operator, measurement, evaluator=None, record=False, verbose=False, **kwargs):
        """
            Samples using the DAPS method.

            Parameters:
                model (nn.Module): (Latent) Diffusion model
                x_start (torch.Tensor): Initial state.
                operator (nn.Module): Operator module.
                measurement (torch.Tensor): Measurement tensor.
                evaluator (Evaluator): Evaluation function.
                record (bool): Whether to record the trajectory.
                verbose (bool): Whether to display progress bar.
                **kwargs:
                    gt (torch.Tensor): reference ground truth data, only for evaluation

            Returns:
                torch.Tensor: The final sampled state.
        """
        if record:
            self.trajectory = Trajectory()
       
        sampler = DPSSampler(self.algo_param, self.diffusion_scheduler)
        x0 = sampler.sample(model, x_start, operator, measurement, SDE=False, record=False, verbose=False)
        if record:
            self.trajectory = sampler.trajectory

        return x0

    def _record(self, xt, x0y, x0hat, sigma, x0hat_results, x0y_results):
        """
            Records the intermediate states during sampling.
        """
        self.trajectory.add_tensor(f'xt', xt)
        self.trajectory.add_tensor(f'x0y', x0y)
        self.trajectory.add_tensor(f'x0hat', x0hat)
        self.trajectory.add_value(f'sigma', sigma)
        for name in x0hat_results.keys():
            self.trajectory.add_value(f'x0hat_{name}', x0hat_results[name])
        for name in x0y_results.keys():
            self.trajectory.add_value(f'x0y_{name}', x0y_results[name])

    def get_start(self, ref):
        """
            Generates a random initial state based on the reference tensor.

            Parameters:
                ref (torch.Tensor): Reference tensor for shape and device.

            Returns:
                torch.Tensor: Initial random state.
        """
        x_start = torch.randn_like(ref) * self.diffusion_scheduler.sigma_max
        return x_start


class FIGSampler(nn.Module):
    """
        FIG sampler for reverse SDE or PF-ODE
    """

    def __init__(self, param, scheduler, solver='euler'):
        """
            Initializes the diffusion sampler with the given scheduler and solver.

            Parameters:
                scheduler (Scheduler): Scheduler instance for managing sigma and timesteps.
                solver (str): Solver method ('euler').
        """
        super().__init__()
        self.param = param
        self.scheduler = scheduler
        self.solver = solver

    def sample(self, model, x_start, operator, measurement, SDE=False, record=False, verbose=False, **kwargs):
        """
            Samples from the diffusion process using the specified model.

            Parameters:
                model (DiffusionModel): Diffusion model supports 'score' and 'tweedie'
                x_start (torch.Tensor): Initial state.
                operator (Operator): Operator module.
                measurement (torch.Tensor): Measurement tensor.
                K (int): correction steps.
                lr (float): gradient descent learning rate 
                SDE (bool): Whether to use Stochastic Differential Equations.
                record (bool): Whether to record the trajectory.
                verbose (bool): Whether to display progress bar.

            Returns:
                torch.Tensor: The final sampled state.
        """
        if self.solver == 'euler':
            return self._euler(model, x_start, operator, measurement, SDE, record, verbose)
        else:
            raise NotImplementedError

    def _euler(self, model, x_start, operator, measurement, SDE=False, record=False, verbose=False):
        """
            Euler's method for sampling from the diffusion process.
        """
        if record:
            self.trajectory = Trajectory()
        pbar = tqdm.tqdm(range(self.scheduler.num_steps)) 
        
        x = x_start
        K = self.param.get('K', 5)
        lr = self.param.get('lr', 0.08)
        mix_coef = self.param.get('mix_coef', 1)
        for step in pbar:
            sigma, factor = self.scheduler.sigma_steps[step], self.scheduler.factor_steps[step]
            score = model.score(x, sigma)
            x0hat = x + score * sigma ** 2
            xt = x0hat + sigma * torch.randn_like(x0hat)

            # step_size = 10
            # if step <= self.scheduler.num_steps - step_size:
            #     y_step = (step // step_size + 1) * step_size
            # else:
            #     y_step = self.scheduler.num_steps - 1
            # y = measurement + self.scheduler.sigma_steps[y_step] * torch.randn_like(measurement)
            y = measurement + sigma * torch.randn_like(measurement)

            if step > 5:
                x.requires_grad_(True)
                for k in range(K):
                    norm = operator.error(x,y).sum()
                    norm_grad = torch.autograd.grad(outputs=norm, inputs=x)[0]
                    x = x - norm_grad  * lr
                x = x.detach()
                if operator.name == 'inpainting':
                    # masked parts + (1-m) * unmasked parts + m * unmasked parts from tweedie
                    x = operator(x) + (1-mix_coef) * (x-operator(x)) + mix_coef * (xt-operator(xt))

            if SDE:
                epsilon = torch.randn_like(x)
                x = x + factor * score + np.sqrt(factor) * epsilon
            else:
                x = x + factor * score * 0.5
            # record
            if record:
                if SDE:
                    self._record(x, score, sigma, factor, epsilon)
                else:
                    self._record(x, score, sigma, factor)

            pbar.set_postfix({'dist': operator.error(x,y).sum()}, refresh=False)
            # torchvision.utils.save_image(y.clamp(-1,1) / 2 + 0.5, f"./tmp/y_{step}.png")
            # torchvision.utils.save_image(x.clamp(-1,1) / 2 + 0.5, f"./tmp/x_{step}.png")

        return x

    def _record(self, x, score, sigma, factor, epsilon=None):
        """
            Records the intermediate states during sampling.
        """
        self.trajectory.add_tensor(f'xt', x)
        self.trajectory.add_tensor(f'score', score)
        self.trajectory.add_value(f'sigma', sigma)
        self.trajectory.add_value(f'factor', factor)
        if epsilon is not None:
            self.trajectory.add_tensor(f'epsilon', epsilon)

    def get_start(self, ref):
        """
            Generates a random initial state based on the reference tensor.

            Parameters:
                ref (torch.Tensor): Reference tensor for shape and device.

            Returns:
                torch.Tensor: Initial random state.
        """
        x_start = torch.randn_like(ref) * self.scheduler.sigma_max
        return x_start


class FIG(nn.Module):
    """
        Implementation of decoupled annealing posterior sampling.
    """

    def __init__(self, algo_param, diffusion_scheduler_config, **kwargs):
        """
            Initializes the DAPS sampler with the given configurations.

            Parameters:
                annealing_scheduler_config (dict): Configuration for annealing scheduler.
                diffusion_scheduler_config (dict): Configuration for diffusion scheduler.
                lgvd_config (dict): Configuration for Langevin dynamics.
        """
        super().__init__()
        self.algo_param = algo_param
        self.diffusion_scheduler = Scheduler(**diffusion_scheduler_config)

    def sample(self, model, x_start, operator, measurement, evaluator=None, record=False, verbose=False, **kwargs):
        """
            Samples using the DAPS method.

            Parameters:
                model (nn.Module): (Latent) Diffusion model
                x_start (torch.Tensor): Initial state.
                operator (nn.Module): Operator module.
                measurement (torch.Tensor): Measurement tensor.
                evaluator (Evaluator): Evaluation function.
                record (bool): Whether to record the trajectory.
                verbose (bool): Whether to display progress bar.
                **kwargs:
                    gt (torch.Tensor): reference ground truth data, only for evaluation

            Returns:
                torch.Tensor: The final sampled state.
        """
       
        sampler = FIGSampler(self.algo_param, self.diffusion_scheduler)
        x0 = sampler.sample(model, x_start, operator, measurement, SDE=False, record=False, verbose=False)
        if record:
            self.trajectory = sampler.trajectory

        return x0

    def _record(self, xt, x0y, x0hat, sigma, x0hat_results, x0y_results):
        """
            Records the intermediate states during sampling.
        """
        self.trajectory.add_tensor(f'xt', xt)
        self.trajectory.add_tensor(f'x0y', x0y)
        self.trajectory.add_tensor(f'x0hat', x0hat)
        self.trajectory.add_value(f'sigma', sigma)
        for name in x0hat_results.keys():
            self.trajectory.add_value(f'x0hat_{name}', x0hat_results[name])
        for name in x0y_results.keys():
            self.trajectory.add_value(f'x0y_{name}', x0y_results[name])

    def get_start(self, ref):
        """
            Generates a random initial state based on the reference tensor.

            Parameters:
                ref (torch.Tensor): Reference tensor for shape and device.

            Returns:
                torch.Tensor: Initial random state.
        """
        x_start = torch.randn_like(ref) * self.diffusion_scheduler.sigma_max
        return x_start

