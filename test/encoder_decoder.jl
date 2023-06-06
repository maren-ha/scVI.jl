using scVI 
# test encoder 
encoder = scVI.scEncoder(100, 100, n_hidden=[128, 64, 32], n_layers=2, distribution = :ln)
@test encoder.n_layers == 2
@test encoder.n_hidden == 128
y = randn(10)
@test all(encoder.z_transformation(y) .== Flux.softmax(y, dims=1))
encoder = scVI.scEncoder(100, 100, n_hidden=[128, 64, 32], n_layers=3, distribution = :something_random)
@test encoder.z_transformation == identity

encoder = scVI.scAEncoder(100, 100, n_hidden=[128, 64, 32], n_layers=2, distribution = :ln)
@test encoder.n_layers == 2
@test encoder.n_hidden == 128
y = randn(10)
@test all(encoder.z_transformation(y) .== softmax(y, dims=1))
encoder = scVI.scAEncoder(100, 100, n_hidden=[128, 64, 32], n_layers=3, distribution = :something_random)
@test encoder.z_transformation == identity

x = randn(Float32, (100,1))
qm, latent = encoder(x)
@test isa(qm, Matrix{Float32})
@test size(qm) == (100, 1)

# test decoder 
decoder = scVI.scDecoder(100, 100, n_hidden=[128, 64, 32], n_layers=2, dispersion = :something_else)
@test decoder.n_layers == 2
@test decoder.n_hidden == 128
@test isa(decoder.px_r_decoder, Vector{Float32})
@test length(decoder.px_r_decoder) == 100