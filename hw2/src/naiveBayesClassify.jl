include("../../common/interface.jl")

function likelihood(XTrain, labels, nClass)
    (nTrain, f) = size(XTrain)
    sumX = zeros(f, nClass)
    sumX2 = zeros(f, nClass)
    countClass = zeros(f, nClass)
    for j in 1:f
        for i in 1:nTrain
            sumX[j, labels[i]] += XTrain[i, j]
            sumX2[j, labels[i]] += XTrain[i, j] ^ 2
            countClass[j, labels[i]] += 1
        end
    end
    # count should not contain zeroes, since nClass is calculated from class
    ex = sumX ./ countClass
    ex2 = sumX2 ./ countClass
    std = sqrt.(ex2 - ex.^2)
    (ex, std)
end

function prior(nClass)
    ones(nClass) ./ nClass
end

function normalpdf(x, mu, sigma)
    return 1 / (sigma * sqrt(2 * pi)) * exp(-0.5 * ((x - mu) / sigma)^2)
end

function relabel(labels)
    u = unique(labels)
    (indexin(labels, u), length(u))
end

function naive_bayes_classify(XTrain, yTrain, XTest)
    (labels, nClass) = relabel(yTrain)
    pri = prior(nClass)
    (mu, sigma) = likelihood(XTrain, labels, nClass)
    (f, nClass) = size(mu)
    (nTest, _f) = size(XTest)
    post = zeros(nTest, nClass)
    for test in 1:nTest
        for k in 1:nClass
            likeClassK = 1
            for fi in 1:f
                likeClassK = likeClassK * normalpdf(XTest[test, fi], mu[fi, k], sigma[fi, k])
            end
            post[test, k] = pri[k] * likeClassK
        end
    end
    yTest = argmax.(eachrow(post))
    order = unique(yTrain)
    map(i -> order[i], yTest)
end

struct NaiveBayesClassifier <: Classifier
    μ::Array{AbstractFloat, 2}
    σ::Array{AbstractFloat, 2}
end
