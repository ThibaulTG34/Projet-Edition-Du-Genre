# <div align=center> Projet édition du genre d'un portrait </div>

## Courte présentation du sujet
L'objectif de ce projet est de développer une interface dans laquelle on pourra importer une image et changer le genre de la personne présente sur l'image.

## Méthodes
D'abord, nous avions implémenté une méthode sans utiliser de réseaux de neurone, utilisant le plaquage de portrait et le lissage. Avec cette méthode, il faut deux portraits relativement proche, car sinon le résultat du plaquage sera trop brutal (voir image ci-après), donc il nous a fallut implémenter des fonctions de distances permettant de trouver le portrait le plus "proche" dans une base de données. 

<p align="center">
<img src="https://github.com/ThibaulTG34/Projet-Edition-Du-Genre/blob/interface/Code/Genre/result.png" alt="image" style="width:300px;height:auto;">
</p>

Ensuite, nous avions implementé une méthode avec des réseaux de neurones, comme par exemple les GANs. Pour le moment, nous avons vu que le CycleGAN revenait souvent pour ce type de projet. Nous avons donc commencé sur cette voie là.
