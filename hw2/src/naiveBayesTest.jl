include("naiveBayesClassify.jl")
include("../../common/dataset.jl")

const datapath = joinpath(@__DIR__, "../data/")

function readdata(dataset_name)
    XTrain, yTrain, XTest, yTest = openall(datapath, dataset_name)
    yTrain = trunc.(Int, yTrain)
    yTest = trunc.(Int, yTest)
    XTrain, yTrain, XTest, yTest
end

function testbayes(dataset_name)
    XTrain, yTrain, XTest, yTest = readdata(dataset_name)
    yTest_bayes = naive_bayes_classify(XTrain, yTrain, XTest)
    sum(yTest .== yTest_bayes) / length(yTest)
end
