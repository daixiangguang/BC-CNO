function partition_best=balanced_kmeans(X,k)
    

% number of points
n = size(X,1);

% number of clusters

% minimum size of a cluster

minimum_size_of_a_cluster = floor(n/k);


% dimensionality

d = size(X,2);

MSE_best = 0; % dummy value

number_of_iterations_distribution = zeros(100,1);

for repeats = 1:1      % 1:100

% initial centroids

for j = 1:k
pass = 0;
while pass == 0
    i = randi(n);
    pass = 1;
    for l = 1:j-1
       if X(i,:) == C(l,:) pass = 0;
       end
    end
end
C(j,:) = X(i,:);
end


partition = zeros(n,1);                 % dummy value
partition_previous = -1;       % dummy value
partition_changed = 1;

kmeans_iteration_number = 0;

while ((partition_changed)&&(kmeans_iteration_number<100))% kmeans iterations
    
partition_previous = partition;

% kmeans assignment step

% setting cost matrix for Hungarian algorithm

costMat = zeros(n);
parfor i=1:n
    for j = 1:n
        costMat(i,j) = (X(j,:)-C(mod(i,k)+1,:))*(X(j,:)-C(mod(i,k)+1,:))';
    end
end

% Execute Hungarian algorithm
[assignment,cost] = munkres(costMat);

% zero partitioning
for i = 1:n
    partition(i) = 0;
end

% find current partitioning from hungarian algorithm result
for i = 1:n 
    if assignment(i) ~= 0
            partition(assignment(i))=mod(i,k)+1;
    end
end

% kmeans update step

for j = 1:k
C(j,:) = mean(X((partition==j),:));
end


kmeans_iteration_number = kmeans_iteration_number +1;

partition_changed = sum(partition~=partition_previous);

end  % kmeans iterations


MSE = 0;
for i = 1:n
    MSE = MSE + ((X(i,:)-C(partition(i),:))*(X(i,:)-C(partition(i),:))')/n;
end

if (MSE<MSE_best)||(repeats==1)
    MSE_best = MSE;
    C_best = C;
    partition_best = partition;
end

MSE_repeats(repeats) = MSE;

number_of_iterations_distribution(kmeans_iteration_number) = number_of_iterations_distribution(kmeans_iteration_number)+1;

end % repeats
    

% new notation

C = C_best;
partition = partition_best;
MSE = MSE_best;

number_of_iterations_distribution;

MSE;

mean_MSE_repeats = mean(MSE_repeats);
std_MSE_repeats = std(MSE_repeats);



       



