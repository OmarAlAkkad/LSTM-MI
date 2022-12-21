import flwr as fl

#this script is used to start a federated learning server using the flwr library.

if __name__ == "__main__":
    # specify the strategy for the server to use
    #Strategies include number of clients required to start round of federated learning,
    #minimum number of clients required to be connected at a given time, and more
    #depending on the use of the current project.
    #More found at https://flower.dev/docs/strategies.html

    #specify number of clients to 2 as each will train on half of the available classes in
    #the mnist datast.
    strategy = fl.server.strategy.FedAvg(
        min_fit_clients=2,  # Minimum number of clients to be sampled for the next round
        min_available_clients=2,  # Minimum number of clients that need to be connected to the server before a training round can start
    )

    fl.server.start_server(
            server_address="127.0.0.1:8500",
            config={"num_rounds": 3}, #specify number of rounds of federated learning to perform
            strategy=strategy,
    )
    #each sript (server/client) should be ran on a seperate terminal for the federated learning to start.