import numpy as np

def ComputeDice(X,Y,verbose=False):
    Xbin = (X > 0)
    Ybin = (Y > 0)
    cardIntersec = np.sum(np.logical_and(Xbin,Ybin))
    cardX = np.sum(Xbin)
    cardY = np.sum(Ybin)

    if verbose:
        print(f"cardIntersec: {cardIntersec}")
        print(f"cardX: {cardX}")
        print(f"cardY: {cardY}")

    if cardY + cardX > 0:
        dice = (2.0*cardIntersec) / (cardX + cardY)
        if verbose:
            print(f"dice: {dice}")
        return dice
    else:
        return 0