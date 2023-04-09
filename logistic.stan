/* References:
- Stan functions reference: 11.2 Ordinary differential equation (ODE) solvers
  https://mc-stan.org/docs/functions-reference/functions-ode-solver.html
- Ordinary Differential Equations with Stan in R
  https://mpopov.com/tutorials/ode-stan-r/
- Predator-Prey Population Dynamics: the Lotka-Volterra model in Stan
  https://mc-stan.org/users/documentation/case-studies/lotka-volterra-predator-prey.html
*/

functions {
  vector logistic(real t, vector x, array[] real theta) {
    vector[1] dx_dt;
    real r = theta[1];
    real K = theta[2];

    dx_dt[1] = r * x[1] * (1 - x[1] / K);
    return dx_dt;
  }
}

data {
  int<lower = 0> N;            // number of measurements
  array[N] real ts;            // measurement times
  real<lower = 0> y0;          // initial measured value
  array[N] real<lower = 0> y;  // measured values
}

parameters {
  real<lower = 0> r;        // intrinsic growth rate
  real<lower = 0> K;        // carrying capacity
  vector<lower = 0>[1] z0;  // initial value
  real<lower = 0> sigma;    // noise scale
}

model {
  array[2] real theta = {r, K};
  array[N] vector[1] z = ode_rk45(logistic, z0, 0, ts, theta);

  y0 ~ lognormal(log(z0), sigma);
  for (n in 1:N)
    y[n] ~ lognormal(log(z[n]), sigma);
  // priors
  r ~ normal(0, 5);
  K ~ normal(0, 100);
  z0 ~ normal(0, 100);
  sigma ~ normal(0, 5);
}
