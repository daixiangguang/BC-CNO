function [obj]=ckmean(data,nClass)
save('plda_bl_score.mat','data')
% clear classes
 obj = py.importlib.import_module('constrained_kmeans');
 py.importlib.reload(obj);
obj=py.constrained_kmeans.main(nClass);