
% this function return all the M^N possiple combination of elements of vector
% s with repetation
% O = allthestates(s) returns a M*N larg matrix O, M = length(s)^p : all
% the possible ways of combination of elements of vector s;
function O = allthestates(s, N)
P = length(s);
O = zeros(P^N, N);

for i = 1:N
    t1 = repmat(s, P^(i-1), 1);
    t2 = reshape(t1, 1, []);
    t3 = repmat(t2, 1, P^(N-i));
    O(:, N-i+1) = t3;
end
