@info "testing highly variable gene utils: `check_nonnegative_integers`..."
using scVI
@test scVI.is_nonnegative_integer(0.3) == false 
@test scVI.is_nonnegative_integer(-6) == false
@test scVI.is_nonnegative_integer(6) == true
x = randn(10, 5)
@test scVI.check_nonnegative_integers(x) == false
x = rand(10, 5)
@test scVI.check_nonnegative_integers(x) == false
x = rand(-10:10, (10, 5))
@test scVI.check_nonnegative_integers(x) == false
x = rand(1:10, (10, 5))
@test scVI.check_nonnegative_integers(x) == true
x = Float32.(x)
@test scVI.check_nonnegative_integers(x) == true
