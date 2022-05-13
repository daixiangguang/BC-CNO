function x=label_to_matrix(label,n,p)
x=-ones(n,p);
for i=1:n
    x(i,label(i))=1;
end