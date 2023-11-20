function [z] = sgn (x)
    z = x;
    z(x<0) = -1;
    z(x>0) = 1;
end