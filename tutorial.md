Series: Vector Search with MongoDB Atlas and Amazon SageMaker
Article #1: Build your own Vector Search with MongoDB Atlas and Amazon SageMaker
You've heard about machine learning, models and AI but don't quite know where to start?
You want to search your data semantically?
Are you interested in using Vector Search in your application?

Then you’ve come to the right place!

This series will introduce you to MongoDB Atlas Vector Search, Amazon SageMaker and how to use both together to semantically search your data.

This first part of the series will focus on the architecture of such an application, i.e. the parts you need, how they are connected, and what they do.

The following parts of the series will then dive into the details about how the individual elements presented in this architecture work in detail (Amazon SageMaker in part #2 and MongoDB Atlas Vector Search in part #3) and their actual configuration and implementation. If you are just interested in one of those two implementations, have a quick look at the architecture pictures and then head to the corresponding part of the series. But to get a deep understanding of Vector Search I recommend reading the full series.

Let’s start with Why though: Why should you use MongoDB Atlas Vector Search and Amazon SageMaker?

# Components of your application
In machine learning, an embeddings model is a type of model that learns to represent objects, such as words, sentences, or even entire documents, as vectors in a high-dimensional space. These vectors, called embeddings, capture semantic relationships between the objects.

On the other hand, a [large language model](https://www.mongodb.com/basics/large-language-models), which is a term you might have heard of, is designed to understand and generate human-like text. It learns patterns and relationships within language by processing vast amounts of text data. While it also generates embeddings as an internal representation, the primary goal is to understand and generate coherent text.

Embedding models are often used in tasks like natural language processing (NLP), where understanding semantic relationships is crucial. For example, word embeddings can be used to find similarities between words based on their contextual usage.

In summary, embeddings models focus on representing objects in a meaningful way in a vector space, while large language models are more versatile, handling a wide range of language-related tasks by understanding and generating text.

For our needs in this application an embeddings model is sufficient. In particular we will be using [All MiniLM L6 v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) by Hugging Face.

[Amazon SageMaker](https://aws.amazon.com/sagemaker/) isn't just another AWS service; it's a versatile platform designed by developers, for developers. It empowers us to take control of our machine learning projects with ease. Unlike traditional ML frameworks, SageMaker simplifies the entire ML lifecycle, from data preprocessing to model deployment. As software engineers, we value efficiency, and SageMaker delivers precisely that, allowing us to focus more on crafting intelligent models and less on infrastructure management. It provides a wealth of pre-built algorithms, making it accessible even for those not deep into the machine learning field.

[MongoDB Atlas Vector Search](https://www.mongodb.com/products/platform/atlas-vector-search) is a game-changer for developers like us who appreciate the power of simplicity and efficiency in database operations. Instead of sifting through complex queries and extensive code, Atlas Vector Search provides an intuitive and straightforward way to implement vector-based search functionality. As software engineers, we know how crucial it is to enhance the user experience with lightning-fast and accurate search results. This technology leverages the benefits of advanced vector indexing techniques, making it ideal for projects involving recommendation engines, content similarity, or even gaming-related features. With MongoDB Atlas Vector Search, we can seamlessly integrate vector data into our applications, significantly reducing development time and effort. It's a developer's dream come true – practical, efficient, and designed to make our lives easier in the ever-evolving world of software development.
# Generating and updating embeddings for your data
There are two steps to using Vector Search in your application.

The first step is to actually create vectors (also called embeddings or embedding vectors), as well as update them whenever your data changes. The easiest way to watch for newly inserted and updated data from your server application is to use [MongoDB Atlas triggers](https://www.mongodb.com/docs/atlas/app-services/triggers/) and watch for exactly those two events. The triggers themselves are out of scope of this tutorial but you can find [other great resources about how to set them up in our developer center](https://www.mongodb.com/developer/products/mongodb/atlas-open-ai-review-summary/).

The trigger then executes a script that creates new vectors. This can for example be done via [MongoDB Atlas Functions](https://www.mongodb.com/docs/atlas/app-services/functions/) or as in this diagram using an AWS Lambda. The script itself then uses the Amazon SageMaker endpoint with your desired model deployed via the REST API to create or update a vector in your Atlas database.

The important bit here that makes the usage so easy and the performance so great is that the data and the embeddings are saved inside the same database:

> Data that belongs together gets saved together.

How to actually deploy and prepare this SageMaker endpoint and offer it as a REST service for your application will be discussed in detail in part #2 of this tutorial.

# Querying your data
The other half of your application will be responsible for actually taking in queries to semantically search your data.

Note that a search has to be done using the vectorized version of the query. And the vectorization has to be done with the same model that we used to vectorize the data itself. The same Amazon SageMaker endpoint can of course be used for that.

Therefore whenever a client application sends a request to the server application, two things have to happen.

1. The server application needs to call the REST service that provides the Amazon SageMaker endpoint (see previous section).
2. With the vector received, the server application then needs to execute a search using Vector Search to retrieve the results from the database.

The implementation of how to query Atlas can be found in part #3 of this tutorial.
# Wrapping it up
This short, first part of the series has provided you with an overview of a possible architecture to use Amazon SageMaker and MongoDB Atlas Vector Search to semantically search your data.

Have a look at part #2 if you are interested in how to set up the Amazon SageMaker and part #3 to go into detail about MongoDB Atlas Vector Search.

✅ Sign-up for a free cluster at → <a href="https://mdb.link/free--ZSKyLspT3Q" target="_blank" rel="noreferrer">https://mdb.link/free--ZSKyLspT3Q</a>

✅ Already have an AWS account? Atlas supports paying for usage via the AWS Marketplace (AWS MP) without any upfront commitment - simply <a href="https://aws.amazon.com/marketplace/pp/prodview-pp445qepfdy34?trk=38088f03-4928-4b00-9a39-86781df6ab9b&sc_channel=el" target="_blank" rel="noreferrer">sign up for MongoDB Atlas via AWS Marketplace<a>

✅ Get help on our Community Forums → <a href="https://mdb.link/community--ZSKyLspT3Q" target="_blank" rel="noreferrer">https://mdb.link/community--ZSKyLspT3Q</a>
Article #2: Create your own model endpoint with Amazon SageMaker, AWS Lambda and AWS API Gateway
Welcome back to part #2 of the `Amazon SageMaker + Atlas Vector Search` series. In the previous part I’ve shown you how to set up an architecture that uses both tools to create embeddings for your data and how to use those to then semantically search through your data.

In this part of the series we will look into the actual doing. No more theory!
Part #2 specifically will show you how to create the REST service described in the architecture.

The REST endpoint will serve as the encoder that creates embeddings (vectors) that will then be used in the next part of this series to search through your data semantically.The deployment of the model will be handled by Amazon SageMaker, AWS's all-in-one ML service. We will expose this endpoint using AWS Lambda and AWS API Gateway later on to make it available to the server app.
# Amazon SageMaker
[Amazon SageMaker](https://aws.amazon.com/pm/sagemaker/) is a cloud based machine-learning platform that enables developers to build, train, and deploy machine learning (ML) models for any use case with fully managed infrastructure, tools, and workflows.

To make it easier to get started, Amazon SageMaker JumpStart provides a set of solutions for the most common use cases that can be deployed readily with just a few clicks.
# Getting Started with Amazon SageMaker
Amazon SageMaker JumpStart helps you quickly and easily get started with machine learning. The solutions are fully customizable and support one-click deployment and fine-tuning of more than 150 popular open source models such as natural language processing, object detection, and image classification models.

Popular solutions include:
- Extract & Analyze Data: Automatically extract, process, and analyze documents for more accurate investigation and faster decision-making.
- Fraud Detection: Automate detection of suspicious transactions faster and alert your customers to reduce potential financial loss.
- Churn Prediction: Predict likelihood of customer churn and improve retention by honing in on likely abandoners and taking remedial actions such as promotional offers.
- Personalized Recommendations: Deliver customized, unique experiences to customers to improve customer satisfaction and grow your business rapidly.
# Let's set up a playground for you to try it out!
> Before we start, make sure you choose a region that is supported for `RStudio` (more on that later) and `JumpStart`. You can check both on the Amazon SageMaker pricing page by checking if your desired region appears in the `On-Demand Pricing` list.

On the main page of Amazon SageMaker you'll find the option to `Set up for a single user`. This will set up a domain and a quick start user.



A QuickSetupDomain is basically just a default configuration so that you can get started deploying models and trying out SageMaker. You can customize it later to your needs.

The initial setup only has to be done once, but it might take several minutes. When finished, Amazon SageMaker will notify you about the new domain being ready.

[Amazon SageMaker Domain](https://docs.aws.amazon.com/sagemaker/latest/dg/sm-domain.html) supports Amazon SageMaker machine learning (ML) environments and contain the following:
The domain itself, which holds an AWS EC2 that models will be deployed onto. This inherently contains a list of authorized users; and a variety of security, application, policy, and Amazon Virtual Private Cloud (Amazon VPC) configurations.
The `UserProfile` represents a single user within a Domain that you will be working with.
A `shared space` which consists of a shared JupyterServer application and shared directory. All users within the Domain have access to the same shared space.
An `App` represents an application that supports the reading and execution experience of the user’s notebooks, terminals, and consoles.


Alt: Domain Details in SageMaker after being created

After the creation of the domain and the user, you can launch the SageMaker Studio which will be your platform to interact with SageMaker, your models and deployments for this user.

[Amazon SageMaker Studio](https://docs.aws.amazon.com/sagemaker/latest/dg/studio.html) is a web-based, integrated development environment (IDE) for machine learning that lets you build, train, debug, deploy, and monitor your machine learning models.



Here, we’ll go ahead and start with a new JumpStart solution.



All you need to do to set up your JumpStart solution is to choose a model. For this tutorial we will be using an embedding model called `All MiniLM L6 v2` by Hugging Face.



When choosing the model, click on `Deploy` and SageMaker will get everything ready for you.



You can adjust the endpoint to your needs but for this tutorial you can totally go with the defaults.



As soon as the model shows its status as `In service`, everything is ready to be used.



Note that the endpoint name here is `jumpstart-dft-hf-textembedding-all-20240117-062453`. Note down your endpoint name, you will need it in the next step.
# Using the model to create embeddings
Now that the model is set up and the endpoint ready to be used, we can expose it for our server application.

We won’t be exposing the SageMaker endpoint directly, instead we will be using AWS API Gateway and AWS Lambda.

Let’s first start by creating the Lambda function that uses the endpoint to create embeddings.

AWS Lambda is an event-driven, serverless computing platform provided by Amazon as a part of Amazon Web Services. It is designed to enable developers to run code without provisioning or managing servers. It executes code in response to events and automatically manages the computing resources required by that codew

In the main AWS Console, go to `AWS Lambda` and click `Create function`.



Choose to `Author from scratch`, give your function a name (`sageMakerLambda` for example) and choose the runtime. For this example, we’ll be running on Python.



When everything is set correctly, create the function.



The following code snippet assumes that the Lambda function and the Amazon SageMaker endpoint are deployed in the same AWS account. All you have to do is to replace `<YOUR_ENDPOINT_NAME>` with your actual endpoint name from the previous section.

Note that the `lambda_handler` returns a status code and a body. It’s ready to be exposed as an endpoint, for using AWS API Gateway.

import json
import boto3

sagemaker_runtime_client = boto3.client("sagemaker-runtime")

def lambda_handler(event, context):
    try:
        # Extract the query parameter 'query' from the event
        query_param = event.get('queryStringParameters', {}).get('query', '')

        if query_param:
            embedding = get_embedding(query_param)
            return {
                'statusCode': 200,
                'body': json.dumps({'embedding': embedding})
            }
        else:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'No query parameter provided'})
            }

    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }

def get_embedding(synopsis):
    input_data = {"text_inputs": synopsis}
    response = sagemaker_runtime_client.invoke_endpoint(
        EndpointName="<YOUR_ENDPOINT_NAME>",
        Body=json.dumps(input_data),
        ContentType="application/json"
    )
    result = json.loads(response["Body"].read().decode())
    embedding = result["embedding"][0]
    return embedding

Don’t forget to click `Deploy`!



One last thing we need to do before we can use this lambda function is to make sure it actually has permissions to execute the SageMaker endpoint. Head to the `Configuration` part of your lambda function and then to `Permissions`. You can just click on the `Role Name` link to get to the associated role in AWS Identity and Access Management(IAM).



In IAM you want to choose `Add permissions`.



You can choose `Attach policies` to attach pre-created policies from the IAM policy list.



For now, let’s use the `AmazonSageMakerFullAccess` but keep in mind to select only those permissions that you need for your specific application.



# Exposing your Lambda function via AWS API Gateway
Now, let’s head to AWS API Gateway, click `Create API` and then `Build` on the `REST API`.



Choose to create a new API and name it, here in the example: `sageMakerApi`.



That’s all you have to do for now, the API endpoint type can stay on regional assuming you created the Lambda function in the same region. Hit `Create API`.



First, we need to create a new resource.



The resource path will be `/`, pick a name like `sageMakerResource`.



Next, you'll get back to your API overview, this time click `Create method`. We need a GET method that integrates with a Lambda function.



Check the `Lambda proxy integration` and choose your lambda function that you created in the previous section. Then create the method.



Finally, don’t forget to deploy the API.



Choose a stage, this will influence the URL that we need to use (API Gateway will show you the full URL in a moment). Since we’re still testing, `TEST` might be a good choice.


This is only a test for a tutorial, but before deploying to production please also add security layers like API keys. When everything is ready, the `Resources` tab should look something like this.



When sending requests to the API Gateway we will receive the query as a URL query string parameter. The next step is to configure API Gateway and tell it so. And also tell it what to do with it.
Go to your `Resources`, click on `GET` again and head to the `Method request` tab. Click `Edit`.



In the `URL query string parameters` section you want to add a new query string by giving it a name, we chose `query` here. Set it to `Required` but not cached and save it.



The new endpoint is created. At this point we can grab the URL and test it via cURL to see if that part worked fine. You can find the full URL (including stage and endpoint) in the `Stages` tab by opening the stage and endpoint and clicking on `GET`. For this example it’s `https://4ug2td0e44.execute-api.ap-northeast-2.amazonaws.com/TEST/sageMakerResource`, you URL should look similar.



Using the Amazon Cloud Shell or any other terminal, try to execute a cURL request:

curl -X GET 'https://4ug2td0e44.execute-api.ap-northeast-2.amazonaws.com/TEST/sageMakerResource?query=foo'

If everything was set up correctly you should get a result that looks like this (the array contains 384 entries in total):

{"embedding": [0.01623343490064144, -0.007662375457584858, 0.01860642433166504, 0.031969036906957626,................... -0.031003709882497787, 0.008777940645813942]}

Your embeddings REST service is ready. Congratulations! Now you can convert your data into a vector with 384 dimensions!

In the next and final part of the tutorial we will be looking into using this endpoint to prepare vectors and execute a vector search using MongoDB Atlas.

✅ Sign-up for a free cluster at → <a href="https://mdb.link/free--ZSKyLspT3Q" target="_blank" rel="noreferrer">https://mdb.link/free--ZSKyLspT3Q</a>

✅ Already have an AWS account? Atlas supports paying for usage via the AWS Marketplace (AWS MP) without any upfront commitment - simply <a href="https://aws.amazon.com/marketplace/pp/prodview-pp445qepfdy34?trk=38088f03-4928-4b00-9a39-86781df6ab9b&sc_channel=el" target="_blank" rel="noreferrer">sign up for MongoDB Atlas via AWS Marketplace<a>

✅ Get help on our Community Forums → <a href="https://mdb.link/community--ZSKyLspT3Q" target="_blank" rel="noreferrer">https://mdb.link/community--ZSKyLspT3Q</a>
Article #3: Semantically search your data with MongoDB Atlas Vector Search
This final part of the series will show you how to use the Amazon SageMaker endpoint created in the previous part and perform a semantic search on your data using MongoDB Atlas Vector Search.The two parts shown in this tutorial will be:

Creating and updating embeddings/vectors for your data
Creating vectors for a search query and sending them via Atlas Vector Search
# Creating a MongoDB cluster and loading the sample data
If you haven’t done so, create a new cluster in your MongoDB Atlas account. Make sure to check `Add sample dataset` to get the sample data we will be working with right away into your cluster.



If you are using a cluster that has already been set up, click on the three dots in your cluster and then `Load Sample Dataset`.

# Creating a Vector Search index
There is one more step we need to take in Atlas, which is creating a search index, specifically for Vector Search.

In your database overview, click on `Create Index`.



The Search page will be shown, click on `Create Search Index` here.

Then choose `Atlas Vector Search` -> `JSON Editor`.



Open the `sample_mflix` database and choose the `embedded_movies` collection (not `movies`). Name your index (here `VectorSearchIndex`).




The configuration for the index needs to state the number of dimensions. That’s depending on the model and in our case it’s 384. The path tells the index which field will be used to store the vectors, we’ll call it `embedding` here. The similarity for text is usually done with the `cosine` function.

{
  "fields": [
    {
      "numDimensions": 384,
      "path": "embedding",
      "similarity": "cosine",
      "type": "vector"
    }
  ]
}

Hit `Create` and you’re good to go.
# Preparing embeddings
Are you ready for the final part?

Let’s have a look at the code (here in Python)!

You can find the full repository on GitHub:
https://github.com/mongodb-developer/AtlasVectorSearchWithAmazonSageMaker

In the following we will look at the three relevant files that show you how you can implement a server app that uses the Amazon SageMaker endpoint.
## Accessing the endpoint: sagemaker.py
The `sagemaker.py` module is the wrapper around the Lambda/Gateway endpoint that we created in the previous example.

Make sure to create a `.env` file with the URL saved in `EMBDDING_SERVICE`.

import os
from typing import Optional
from urllib.parse import quote

import requests
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_SERVICE = os.environ.get("EMBEDDING_SERVICE")

This function will then attach the query that we want to search for to the URL and execute it.

As a result we expect to find the vector in a JSON field called `embedding`.

def create_embedding(plot: str) -> Optional[float]:
    encoded_plot = quote(plot)
    embedding_url = f"{EMBEDDING_SERVICE}?query={encoded_plot}"

    embedding_response = requests.get(embedding_url)
    embedding_vector = embedding_response.json()["embedding"]

    return embedding_vector

## Access and searching the data: atlas.py
The module `atlas.py` is the wrapper around everything MongoDB Atlas.

Similar to `sagemaker.py` we first grab the `MONGODB_CONNECTION_STRING` that you can retrieve in Atlas by clicking on `Connect` in your cluster. It’s the authenticated URL to your cluster.

We then go ahead and define a bunch of variables that we’ve set in earlier parts, like `VectorSearchIndex` and `embedding` along with the automatically created `sample_mflix` demo data.

Using the Atlas driver for Python (called PyMongo) we then create a `MongoClient` which holds the connection to the Atlas cluster.

import os

from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne

from sagemaker import create_embedding

load_dotenv()

MONGODB_CONNECTION_STRING = os.environ.get("MONGODB_CONNECTION_STRING")
DATABASE_NAME = "sample_mflix"
COLLECTION_NAME = "embedded_movies"
VECTOR_SEARCH_INDEX_NAME = "VectorSearchIndex"
EMBEDDING_PATH = "embedding"
mongo_client = MongoClient(MONGODB_CONNECTION_STRING)
database = mongo_client[DATABASE_NAME]
movies_collection = database[COLLECTION_NAME]

The first step will be to actually prepare the already existing data with embeddings.

This is the sole purpose of the `add_missing_embeddings` function.

We’ll create a filter for the documents with missing embeddings and retrieve those from the database, only showing their plot, which is the only field we’re interested in for now.

Assuming we will only find a couple every time, we can then go through them and call the `create_embedding` endpoint for each, creating an embedding for the plot of the movie.

We’ll then add those new embeddings to the `movies_to_update` array so that we eventually only need one `bulk_write` to the database, which makes the call more efficient.

Note that for huge datasets with many embeddings to create, you might want to adjust the lambda function to take an array of queries instead of just a single query. For this simple example it will do.

def add_missing_embeddings():
    movies_with_a_plot_without_embedding_filter = {
        "$and": [
            {"plot": {"$exists": True, "$ne": ""}},
            {"embedding": {"$exists": False}},
        ]
    }
    only_show_plot_projection = {"plot": 1}

    movies = movies_collection.find(
        movies_with_a_plot_without_embedding_filter,
        only_show_plot_projection,
    )

    movies_to_update = []

    for movie in movies:
        embedding = create_embedding(movie["plot"])
        update_operation = UpdateOne(
            {"_id": movie["_id"]},
            {"$set": {"embedding": embedding}},
        )
        movies_to_update.append(update_operation)

    if movies_to_update:
        result = movies_collection.bulk_write(movies_to_update)
        print(f"Updated {result.modified_count} movies")

    else:
        print("No movies to update")


Now that the data is prepared, we add two more functions that we need to offer a nice REST service for our client application.

First, we want to be able to update the plot, which inherently needs to update the embeddings again.

The `update_plot` is similar to the initial `add_missing_embeddings` function but a bit simpler since we only need to update one document.

def update_plot(title: str, plot: str) -> dict:
    embedding = create_embedding(plot)

    result = movies_collection.find_one_and_update(
        {"title": title},
        {"$set": {"plot": plot, "embedding": embedding}},
        return_document=True,
    )

    return result


The other function we need to offer is the actual vector search. This can be done using the [MongoDB Atlas Aggregation pipeline](https://www.mongodb.com/docs/manual/core/aggregation-pipeline/) that can be accessed via the [Atlas Driver](https://www.mongodb.com/docs/atlas/driver-connection/).

The `$vectorSearch` stage needs to include the index name we want to use, the path to the embedding along with the information about how many results we want to get. This time we only want to retrieve the title, so we add a `$project` stage to the pipeline. Make sure to use `list` to turn the cursor that the search returns into a python list.

def execute_vector_search(vector: [float]) -> list[dict]:
    vector_search_query = {
        "$vectorSearch": {
            "index": VECTOR_SEARCH_INDEX_NAME,
            "path": EMBEDDING_PATH,
            "queryVector": vector,
            "numCandidates": 10,
            "limit": 5,
        }
    }
    projection = {"$project": {"_id": 0, "title": 1}}
    results = movies_collection.aggregate([vector_search_query, projection])
    results_list = list(results)

    return results_list
## Putting it all together: main.py
Now we can put it all together. Let’s use flask to expose a REST service for our client application.

from flask import Flask, request, jsonify

from atlas import execute_vector_search, update_plot
from sagemaker import create_embedding

app = Flask(__name__)

One route we want to expose is `/movies/<title>` that can be executed with a `PUT` operation to update the plot of a movie given the title. The title will be a query parameter while the plot is passed in via the body. This function is using the `update_plot` that we created before in `atlas.py` and returns the movie with its new plot on success.

@app.route("/movies/<title>", methods=["PUT"])
def update_movie(title: str):
    try:
        request_json = request.get_json()
        plot = request_json["plot"]
        updated_movie = update_plot(title, plot)

        if updated_movie:
            return jsonify(
                {
                    "message": "Movie updated successfully",
                    "updated_movie": updated_movie,
                }
            )
        else:
            return jsonify({"error": f"Movie with title {title} not found"}), 404

    except Exception as e:
        return jsonify({"error": str(e)}), 500

The other endpoint, finally, is the vector search: `/movies/search`.

A `query` is `POST`’ed to this endpoint which will then use `create_embedding` first to create a vector from this query. Note that we need to also create vectors for the query because that’s what the vector search needs to compare it to the actual data (or rather, its embeddings).

We then call `execute_vector_search` with this `embedding` to retrieve the results, which will be returned on success.

@app.route("/movies/search", methods=["POST"])
def search_movies():
    try:
        request_json = request.get_json()
        query = request_json["query"]
        embedding = create_embedding(query)

        results = execute_vector_search(embedding)

        jsonified_results = jsonify(
            {
                "message": "Movies searched successfully",
                "results": results,
            }
        )

        return jsonified_results

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)

And that’s about all you have to do. Easy, wasn’t it?

Go ahead and run the flask app (main.py) and when ready, send a cURL to see Atlas Vector Search in action. Here is an example when running it locally:


curl -X POST -H "Content-Type: application/json" -d '{"query": "A movie about the Earth, Mars and an invasion."}' http://127.0.0.1:5000/movies/search

This should lead to the following result:


{
  "message": "Movies searched successfully",
  "results": [
    {
      "title": "The War of the Worlds"
    },
    {
      "title": "The 6th Day"
    },
    {
      "title": "Pixels"
    },
    {
      "title": "Journey to Saturn"
    },
    {
      "title": "Moonraker"
    }
  ]
}

“The War of the Worlds” - a movie about Earth, Mars and an invasion. And what a great one, right?
# That’s a wrap!
Of course this is just a quick and short overview of how to use Amazon SageMaker to create vectors and then search via Vector Search.

We do have a full workshop for you to learn about all those parts in detail, please visit the [Search Lab GitHub page](https://github.com/mongodb-developer/search-lab) to learn more.

✅ Sign-up for a free cluster at → <a href="https://mdb.link/free--ZSKyLspT3Q" target="_blank" rel="noreferrer">https://mdb.link/free--ZSKyLspT3Q</a>

✅ Already have an AWS account? Atlas supports paying for usage via the AWS Marketplace (AWS MP) without any upfront commitment - simply <a href="https://aws.amazon.com/marketplace/pp/prodview-pp445qepfdy34?trk=38088f03-4928-4b00-9a39-86781df6ab9b&sc_channel=el" target="_blank" rel="noreferrer">sign up for MongoDB Atlas via AWS Marketplace<a>

✅ Get help on our Community Forums → <a href="https://mdb.link/community--ZSKyLspT3Q" target="_blank" rel="noreferrer">https://mdb.link/community--ZSKyLspT3Q</a>
