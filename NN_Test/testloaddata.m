clear;
data = loadjson('train.json');
num_tests = size(data, 2);

features = {'request_id' 'request_title'};
X = [];
placeholder_cell = cell(1);

for i=1:8
    holder = [];
    temp = data(i);
    temp = temp{:};
    for k=1:size(features, 2)
        feature = features(k);
        getfield(temp, feature{:});
        placeholder_cell{1} = uint8(getfield(temp, feature{:}));
        holder = [holder, placeholder_cell];
    end
    X = [X; holder];
end