include("../hw2/src/naiveBayesClassify.jl")
include("interface.jl")

using MAT
using Statistics

function openpart(dir, dataset_name, part)
    file = matopen(joinpath(dir, dataset_name, "$part$dataset_name.mat"))
    read(file, part)
end

function openset(dir, dataset_name, parts)
    map(part -> openpart(dir, dataset_name, part), parts)
end

function opentrain(dir, dataset_name)
    openset(dir, dataset_name, ["XTrain"; "yTrain"])
end

function opentest(dir, dataset_name)
    openset(dir, dataset_name, ["XTest"; "yTest"])
end

function openall(dir, dataset_name)
    openset(dir, dataset_name, ["XTrain"; "yTrain"; "XTest"; "yTest"])
end

function kfolds(n; k = 5)
    k <= n || throw(ArgumentError("cannot create more folds than observations"))
    sizes = fill(floor(Int, n / k), k)
    for i in 1:(n % k)
        sizes[i] += 1
    end
    offsets = cumsum(sizes) .- sizes .+ 1
    val_indices = map((o, s) -> o:o+s-1, offsets, sizes)
    train_indices = map(indices -> setdiff(1:n, indices), val_indices)
    zip(train_indices, val_indices)
end

function kfolds(X, y; k = 5)
    size(X, 1) == length(y) || throw(ArgumentError("numbers of observations and labels do not match"))
    kfolds(length(y), k=k)
end

function cross_val_score(model, X, y; k = 5)
    scores = zeros(k)
    for (i, (train, val)) in enumerate(kfolds(X, y, k=k))
        prediction = naive_bayes_classify(X[train, :], y[train, :], X[val, :])
        scores[i] = sum(prediction .== y[val, :]) / length(val)
    end
    (scores, mean(scores), std(scores))
end
