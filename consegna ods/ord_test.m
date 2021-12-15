sample = images(:,10);
out2 = @(x) elu(w2*x+b2);
out3 = @(x) elu(w3*out2(x)+b3);
out = @(x) elu(w4*out3(x)+b4);
[m_o, O_class] = max(out(sample));
f = @(x) log(abs(out(x)));
g = @(x) x(O_class);
h = @(x) x([1:(O_class-1), (O_class+1):end]);
th = @(x) (1+tanh(x)/0.9999999)/(2);
% obj_ord = @(x) max(g(f(sample + x) - max(h(f(sample + x)))),0);
obj_ord = @(x) max(g(f(th(x)) - max(h(f(th(x))))),0); %%%%%%!!!!!!!
n = 784; % dimension

% (2) Define an n-by-m matrix of atoms, where each column is an n-dimensional atom
%--------------------------------------------------------------------------
d = length(sample);
atoms = zeros(d, 2*d);
epsilon = 40;  %%!!!
for i = 1:d
    atoms(i,i) = epsilon;
    atoms(i,i + d) = -epsilon; 
    atoms(:,i) = atoms(:,i) + atanh(((2)*sample-1)*0.9999999); %%!!!
    atoms(:,i+d) = atoms(:,i+d) + atanh(((2)*sample-1)*0.9999999);  %%!!!
end
atoms = real(atoms);
%--------------------------------------------------------------------------
% (3) Choose an atom to use as starting point
%--------------------------------------------------------------------------
i0 = randi(2*d); % index of the atom to use as starting point

%--------------------------------------------------------------------------
opts.eps_opt = 1e-7;
opts.n_initial_atoms = 50; %%!!!
opts.verbosity  = true;
% opts.f_stop = 0; %%!!!
% (4) call ORD
[x_ord,y_ord,f_ord,n_f_ord,it_ord,t_elap_ord,flag_ord] = ORD(obj_ord, atoms, i0, opts);

%--------------------------------------------------------------------------
% *** EXAMPLE OF HOW TO CHANGE ORD PARAMETERS ***
% (see the file 'syntax_ord.txt' to know which parameters can be changed
% and their default values)
%
% Instead of calling ORD by the above instruction, do the following:
%
% - create a structure having as field names the names of the parameters
%   to be changed and assign them new values, e.g.,
%
%     opts.verbosity = false;
%
% - pass the structure to ORD as fourth input argument, e.g.,
%
%     [x,y,f,n_f,it,t_elap,flag] = ORD(obj,A,i0,opts);
%--------------------------------------------------------------------------

% write statistics to the screen
fprintf(['\n********************** FINAL RESULTS **********************' ...
         '\nAlgorithm: ORD' ...
         '\nf =  %-.4e'   ...
         '\nobjective function evaluations = %-i' ...
         '\niterations = %-i' ...
         '\ncpu time (s) = %-.2e' ...
         '\nflag = %-i' ...
         '\n***********************************************************\n\n'], ...
         f_ord,n_f_ord,it_ord,t_elap_ord,flag_ord);
new_image =th(x_ord);

% for i = 1:length(new_image)
%     if new_image(i)>1
%         new_image(i) = 1;
%     elseif new_image(i)<0
%         new_image(i) = 0;
%    end
% end

colormap gray
figure(1)
imagesc(reshape(sample, 28,28)')
figure(2)
colormap gray
imagesc(reshape(new_image, 28,28)')
figure(3)
colormap gray
imagesc(reshape(abs(new_image-sample), 28,28)')
[m, N_class] = max(out(new_image));
fprintf(['\nPrevious Class = %-i'...
         '\n New Class = %-i \n\n'], ...
         O_class -1, N_class-1)
     