@info "testing log_nb_positive_julia function..."
mu = rand(10, 2)
theta = rand(10, 2)
x = rand.(NegativeBinomial.(theta, theta ./ (theta .+ mu)))
@test scVI.log_nb_positive_julia(x, mu, theta) ≈ scVI.log_nb_positive(eltype(mu).(x), mu, theta, 1e-15)

@info "testing log_poisson function..."
@test typeof(scVI.log_poisson(eltype(mu).(x), mu)) == typeof(mu)
@test size(scVI.log_poisson(eltype(mu).(x), mu)) == size(x)

@info "testing additional convenience functions for logits conversions..."
total_counts, logits = scVI._convert_mean_disp_to_counts_logits(mu, theta, 1e-15)
new_mu, new_theta = scVI._convert_counts_logits_to_mean_disp(total_counts, logits)
@test new_mu ≈ mu
@test new_theta ≈ theta

new_total_counts, new_logits = scVI._convert_mean_disp_to_counts_logits(new_mu, new_theta, 1e-15)
@test new_total_counts ≈ total_counts
@test new_logits ≈ logits

@info "testing conversion of logits to probabilities..."
probs = scVI.logits_to_probs(logits)
@test all(0 .< probs .< 1)

@info "testing gamma function..."
gamma_distributions = scVI._gamma(theta, mu)
gamma_sample = rand.(gamma_distributions)
@test size(gamma_sample) == size(mu)
@test all(0 .<= gamma_sample)
