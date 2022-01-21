function hc = Ihk(s, reference, paramk, sample, alg)
[~, m] = size(s);
if strcmp(alg, 'Direct-NT') || strcmp(alg, 'Direct-Norm')
    % generate directions for approximation methods in direction space
    if strcmp(alg, 'Direct-NT')
        direction = Hua_direction(m, sample);
    else
        direction = normal_direction(m, sample);
    end
    direction(direction==0) = min(direction(direction~=0))*1e-5; % no great difference
    direction = 1./direction;
    sphere = 2*pi^(m/2)/gamma(m/2)/(2^m);
    hc = Ihk_DirectionC(s, reference, paramk, direction, sphere);
    return
end
if strcmp(alg, 'Polar')
    hc = Ihk_PolarC(s, reference, paramk, sample);
    return
end
