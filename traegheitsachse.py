import numpy as np

#masses
massen_lexikon = {
    'H': 1.00784,
    'C': 12.0096,
    'N': 14.00643,
    'O': 15.99903,
    'F': 18.998403163,
    'P': 30.973761998,
    'S': 32.059,
    'Cl': 35.446,
    'Br': 79.904,
    'I': 126.90447,
    'Li': 6.94,
    'Be': 9.0122,
    'Na': 22.98976928,
    'Mg': 24.305,
    'B' : 10.81,
    'Al': 26.9815385,
}

# Einlesen
read_in_coord = np.genfromtxt('example.xyz', skip_header=2, dtype='str')
natoms = read_in_coord.shape[0]
atoms = read_in_coord[:, 0]
coords = read_in_coord[:, 1:].astype(float)
print (f"Atomtypen: {atoms}")
print (f"Koordinaten: {coords}")

#Zuordnung der Massen
massen = np.array([massen_lexikon[atom] for atom in atoms])
print (f"Massen: {massen}")

# Berechnung des Schwerpunkts
gesamtmasse = np.sum(massen)
zähler = massen @ coords # Matrix-Vektor-Produkt: (1, n) @ (n, 3) = (1, 3) -> Vektor der Länge 3
print(f"Zähler: {zähler}")
schwerpunkt = zähler / gesamtmasse # Vektor der Länge 3 / Skalar = Vektor der Länge 3
print (f"Schwerpunkt: {schwerpunkt}")

#Verschiebung der Koordinaten um den Schwerpunkt
newcoord = coords - schwerpunkt
print (f"Neue Koordinaten: {newcoord}")

#Berechnung des Trägheitstensors: Vektorgeometrisch
def vektorgeometrischer_traegheitstensor(massen, newcoord):
    ri2 = np.sum(newcoord**2, axis=1) # Quadrate der Abstände zum Schwerpunkt
    vorfaktor = np.sum(massen * ri2)
    I =  vorfaktor * np.eye(3) - (newcoord.T * massen) @ newcoord
    I = I.round(10)
    print (f"Trägheitstensor (vektorgeometrisch)a:\n{I}")
    return I

I = vektorgeometrischer_traegheitstensor(massen, newcoord)
IEigenVal, IEigenVec = np.linalg.eigh(I)
print (f"Eigenwerte: {IEigenVal}")
print (f"Eigenvektoren:\n{IEigenVec}")