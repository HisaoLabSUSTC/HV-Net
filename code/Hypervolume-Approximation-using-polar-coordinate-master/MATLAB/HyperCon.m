function hc = HyperCon(s, reference, sample, alg)
[~, m] = size(s);
if strcmp(alg, 'Direct-NT') || strcmp(alg, 'Direct-Norm')
    % generate directions for approximation methods in direction space
    if strcmp(alg, 'Direct-NT')
        direction = Hua_direction(m, sample);
    else
        direction = normal_direction(m, sample);
    end
    direction(direction==0) = min(direction(direction~=0))*1e-5;
    direction = 1./direction;
    sphere = 2*pi^(m/2)/gamma(m/2)/(2^m);
    hc = HC_DirectionC(s, reference, direction, sphere);
    return
end
if strcmp(alg, 'Polar')
    hc = HC_PolarC(s, reference, sample);
    return
end
