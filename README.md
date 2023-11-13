# <div align=center> Projet édition du genre d'un portrait </div>

## Courte présentation du sujet
L'objectif du projet est de développer une interface où l'on pourra importer une image et changer le genre de la personne présente sur l'image.

## Méthodes
D'abord, nous implémenterons une méthode sans utiliser de réseaux de neurone, utilisant le plaquage de portrait et du lissage. Avec cette méthode, il faut deux portraits relativement proche, car sinon le résultat du plaquage sera trop brutal (voir image ci-après), donc il nous a fallut implémenter des fonctions de distances permettant de trouver le portrait le plus "proche" dans une base de données. 

<p align="center">
<img src="https://github.com/ThibaulTG34/Projet-Edition-Du-Genre/blob/interface/Code/Genre/result.png" alt="image" style="width:300px;height:auto;">
</p>

Ensuite, nous implémenterons une méthode avec des réseaux de neurones, comme par exemple les GANs. Pour le moment, nous avons vu que le CycleGAN revenait souvent pour ce type de projet. Nous avons donc commencé sur cette voie là.

## Matériels et Langages
Nous travaillerons principalement sur nos ordinateurs personnels, mais on pourra travailler sur les ordinateurs de la fac lors séances dédiées. Nous développerons le projet en Python et nous utiliserons la bibliothèque Qt pour le développement de toute l'interface.

## Améliorations possibles
- possibilité de voir le changement de genre en direct via la caméra
