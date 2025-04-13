#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script de comparaison de méthodes en Federated Learning appliqué à MNIST :

1. Modèle centralisé (baseline)
2. FedAvg classique (tous les edges participent)
3. FedProx simplifié (avec un terme proximal dans l'entraînement local)

Les résultats (loss et accuracy) sont suivis par round de fédération.
Les courbes d'évolution sont affichées et sauvegardées en PNG et PDF.

Auteur : [Votre Nom]
Date   : [Date]
"""

import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt

# Import des modules locaux (assurez-vous que fl_dataquest.py et fl_model.py sont accessibles)
import fl_dataquest  # pour get_data, get_dataset
import fl_model      # pour MyModel

# =============================================================================
# Fonctions utilitaires pour FedAvg (ces fonctions restent les mêmes)
# =============================================================================

def make_edges(image_list, label_list, num_edges=10, verbose=0):
    """
    Répartit les données (images et labels) entre 'num_edges' clients de façon aléatoire.

    Args:
        image_list (list): liste des images (tableaux NumPy)
        label_list (list): liste des labels (one-hot encodés)
        num_edges (int): nombre de clients
        verbose (int): niveau de verbosité

    Returns:
        dict: un dictionnaire où chaque clé est 'edge_i' et la valeur est un tf.data.Dataset
    """
    edge_names = [f"edge_{i}" for i in range(num_edges)]
    data = list(zip(image_list, label_list))
    random.shuffle(data)
    size = len(data) // num_edges
    shards = [data[i:i + size] for i in range(0, size * num_edges, size)]
    edges = {edge_names[i]: shards[i] for i in range(num_edges)}
    edges_batched = {}
    for en, shard_data in edges.items():
        il, ll = zip(*shard_data)
        il, ll = list(il), list(ll)
        if verbose > 0:
            print(f"{en} => #images = {len(il)}; shapeX = {il[0].shape}")
        ds = tf.data.Dataset.from_tensor_slices((il, ll)).shuffle(len(il)).batch(32)
        edges_batched[en] = ds
    return edges_batched

def weight_scaling_factor(edges, edge_name):
    """
    Calcule la fraction (# d'exemples dans un edge) / (# total d'exemples).
    On suppose ici une taille de batch de 32.
    """
    total_count = sum(len(edges[en]) for en in edges.keys()) * 32
    local_count = len(edges[edge_name]) * 32
    return local_count / total_count

def scale_model_weights(weights, scalar):
    """
    Multiplie chaque tenseur de poids par 'scalar'.
    """
    return [w * scalar for w in weights]

def sum_scaled_weights(scaled_weight_list):
    """
    Agrège les poids (somme layer-wise) des mises à l'échelle.
    Correspond à la moyenne dans FedAvg.
    """
    new_weight = []
    for layers_tuple in zip(*scaled_weight_list):
        layer_sum = tf.math.reduce_sum(layers_tuple, axis=0)
        new_weight.append(layer_sum)
    return new_weight

# =============================================================================
# Implémentation de FedProx (version simplifiée)
# =============================================================================

def local_training_fedprox(model, dataset, local_epochs=1, mu=0.1):
    """
    Entraîne le modèle local avec FedProx en ajoutant un terme proximal.
    Args:
        model : instance de MyModel (modèle local)
        dataset : tf.data.Dataset pour l'entraînement local
        local_epochs (int): nombre d'époques locales
        mu (float): paramètre de régularisation proximal
    """
    # On récupère les poids actuels du modèle global comme référence
    global_weights = model.get_weights()

    # Utiliser Adam ou SGD, ici on reprend SGD pour la cohérence
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

    # Parcours sur local_epochs
    for epoch in range(local_epochs):
        for batch_x, batch_y in dataset:
            with tf.GradientTape() as tape:
                # Prédiction et loss classique
                logits = model.model(batch_x, training=True)
                loss = model.model.compiled_loss(batch_y, logits)
                # Ajout du terme proximal pour chaque couche
                prox_term = 0
                for w_local, w_global in zip(model.get_weights(), global_weights):
                    prox_term += tf.reduce_sum(tf.square(w_local - w_global))
                loss += 0.5 * mu * prox_term  # 0.5 pour standardiser, mu est le paramètre
            gradients = tape.gradient(loss, model.model.trainable_weights)
            optimizer.apply_gradients(zip(gradients, model.model.trainable_weights))
    return model.get_weights()


# =============================================================================
# Expérimentation et visualisation
# =============================================================================

if __name__ == '__main__':
    # Fixer les seeds pour la reproductibilité
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)

    # -------------------------------------------------------------------------
    # a) Chargement et préparation des données MNIST
    # -------------------------------------------------------------------------
    X_train, X_test, y_train, y_test, input_shape = fl_dataquest.get_data(
        img_path='./Mnist/trainingSet/trainingSet/',
        verbose=1
    )
    dtt, dts = fl_dataquest.get_dataset(X_train, X_test, y_train, y_test,
                                        batch_size=32, verbose=1)

    # -------------------------------------------------------------------------
    # b) Entraînement centralisé (baseline)
    # -------------------------------------------------------------------------
    cmodel = fl_model.MyModel(input_shape, nbclasses=10)
    cmodel.fit_it(trains=dtt, epochs=5, tests=dts, verbose=1)
    loss_c, acc_c = cmodel.evaluate(dts, verbose=1)
    print(f"\n[Modèle centralisé] => Loss = {loss_c:.4f}  Acc = {acc_c:.4f}")
    central_results = {"loss": loss_c, "accuracy": acc_c}

    # -------------------------------------------------------------------------
    # c) Création des edges (partitionnement classique)
    # -------------------------------------------------------------------------
    num_edges = 5
    edges_classic = make_edges(X_train, y_train, num_edges=num_edges, verbose=1)

    # -------------------------------------------------------------------------
    # d) Comparaison FedAvg vs FedProx
    # -------------------------------------------------------------------------
    num_rounds = 5
    edge_epoch = 1

    # 1) FedAvg classique (tous les edges participent)
    fedmodel_avg = fl_model.MyModel(input_shape, nbclasses=10)
    fedmodel_avg.set_weights(cmodel.get_weights())
    rounds_avg = []
    fedavg_acc = []
    fedavg_loss = []

    for r in range(num_rounds):
        print(f"\n=== FedAvg - Round {r + 1}/{num_rounds} ===")
        scaled_local_weight_list = []
        for en in edges_classic.keys():
            local_model = fl_model.MyModel(input_shape, nbclasses=10)
            local_model.set_weights(fedmodel_avg.get_weights())
            dtrain_edge = edges_classic[en]
            local_model.fit_it(trains=dtrain_edge, epochs=edge_epoch, tests=None, verbose=0)
            sf = weight_scaling_factor(edges_classic, en)
            w_scaled = scale_model_weights(local_model.get_weights(), sf)
            scaled_local_weight_list.append(w_scaled)
            K.clear_session()
        fedavg_weights = sum_scaled_weights(scaled_local_weight_list)
        fedmodel_avg.set_weights(fedavg_weights)
        loss_avg, acc_avg = fedmodel_avg.evaluate(dts, verbose=0)
        print(f"  FedAvg Round {r + 1}: Loss = {loss_avg:.4f}, Acc = {acc_avg:.4f}")
        rounds_avg.append(r + 1)
        fedavg_loss.append(loss_avg)
        fedavg_acc.append(acc_avg)

    # 2) FedProx simplifié (tous les edges participent)
    fedmodel_prox = fl_model.MyModel(input_shape, nbclasses=10)
    fedmodel_prox.set_weights(cmodel.get_weights())
    rounds_prox = []
    fedprox_acc = []
    fedprox_loss = []
    mu = 0.1  # Paramètre de régularisation proximal

    for r in range(num_rounds):
        print(f"\n=== FedProx - Round {r + 1}/{num_rounds} ===")
        local_weights_list = []
        for en in edges_classic.keys():
            local_model = fl_model.MyModel(input_shape, nbclasses=10)
            local_model.set_weights(fedmodel_prox.get_weights())
            dtrain_edge = edges_classic[en]
            # Utilisation de FedProx pour l'entraînement local
            w_local = local_training_fedprox(local_model, dtrain_edge, local_epochs=edge_epoch, mu=mu)
            sf = weight_scaling_factor(edges_classic, en)
            w_scaled = scale_model_weights(w_local, sf)
            local_weights_list.append(w_scaled)
            K.clear_session()
        fedprox_weights = sum_scaled_weights(local_weights_list)
        fedmodel_prox.set_weights(fedprox_weights)
        loss_prox, acc_prox = fedmodel_prox.evaluate(dts, verbose=0)
        print(f"  FedProx Round {r + 1}: Loss = {loss_prox:.4f}, Acc = {acc_prox:.4f}")
        rounds_prox.append(r + 1)
        fedprox_loss.append(loss_prox)
        fedprox_acc.append(acc_prox)

    # -------------------------------------------------------------------------
    # e) Comparaison finale
    # -------------------------------------------------------------------------
    final_loss_avg, final_acc_avg = fedmodel_avg.evaluate(dts, verbose=1)
    final_loss_prox, final_acc_prox = fedmodel_prox.evaluate(dts, verbose=1)
    print(f"\n[Résultat final FedAvg] => Loss = {final_loss_avg:.4f}, Acc = {final_acc_avg:.4f}")
    print(f"[Résultat final FedProx] => Loss = {final_loss_prox:.4f}, Acc = {final_acc_prox:.4f}")
    print(f"\nComparaison:")
    print(f" - Centralisé       => Acc = {central_results['accuracy']:.4f}")
    print(f" - FedAvg           => Acc = {final_acc_avg:.4f}")
    print(f" - FedProx          => Acc = {final_acc_prox:.4f}")

    # -------------------------------------------------------------------------
    # f) Visualisation des courbes d'évolution des performances
    # -------------------------------------------------------------------------
    plt.figure(figsize=(14, 6))

    # Courbe d'évolution de l'accuracy
    plt.subplot(1, 2, 1)
    plt.plot(rounds_avg, fedavg_acc, marker='o', linestyle='-', color='blue', label='FedAvg')
    plt.plot(rounds_prox, fedprox_acc, marker='o', linestyle='-', color='green', label='FedProx')
    plt.hlines(central_results['accuracy'], rounds_avg[0], rounds_avg[-1],
               colors='red', linestyles='dashed', label='Modèle centralisé')
    plt.title("Évolution de l'Accuracy par Round")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    # Courbe d'évolution de la Loss
    plt.subplot(1, 2, 2)
    plt.plot(rounds_avg, fedavg_loss, marker='o', linestyle='-', color='blue', label='FedAvg')
    plt.plot(rounds_prox, fedprox_loss, marker='o', linestyle='-', color='green', label='FedProx')
    plt.title("Évolution de la Loss par Round")
    plt.xlabel("Round")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.suptitle("Comparaison des performances en FedAvg vs FedProx")
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Sauvegarde des graphiques
    plt.savefig("performance_comparison.png", format="png")
    plt.savefig("performance_comparison.pdf", format="pdf")
    plt.show()
