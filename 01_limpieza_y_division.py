
import argparse
import collections
import datetime
import hashlib
import os
import random
import re
import shutil

import pandas as pd
from PIL import Image

CARPETAS = ["Good Quality_Fruits", "Bad Quality_Fruits"]
GAP_SEG = 10                        
PROPORCIONES = {"train": 0.70, "val": 0.15, "test": 0.15}
SEMILLA = 42

RE_AUMENTADA = re.compile(r"_\d{5}\.jpe?g$", re.IGNORECASE) 
RE_FECHA_NOMBRE = re.compile(r"(\d{8})_?(\d{6})")            
RE_SECUENCIA = re.compile(r"IMG_(\d{4})\.jpe?g$", re.IGNORECASE)  


def fecha_captura(ruta, nombre):
    try:
        dt = Image.open(ruta).getexif().get(306)
        if dt:
            return datetime.datetime.strptime(dt, "%Y:%m:%d %H:%M:%S").timestamp()
    except Exception:
        pass
    m = RE_FECHA_NOMBRE.search(nombre)
    if m:
        try:
            return datetime.datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S").timestamp()
        except ValueError:
            pass
    m = RE_SECUENCIA.match(nombre)
    if m:
        return 1e12 + int(m.group(1))
    return None


def md5(ruta):
    with open(ruta, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def leer_dataset(origen):
    filas = []
    for carpeta in CARPETAS:
        for clase in sorted(os.listdir(os.path.join(origen, carpeta))):
            dir_clase = os.path.join(origen, carpeta, clase)
            if not os.path.isdir(dir_clase):
                continue
            for nombre in sorted(os.listdir(dir_clase)):
                if not nombre.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                fruta, calidad = clase.split("_")
                filas.append({
                    "ruta": os.path.join(carpeta, clase, nombre),
                    "clase": clase, "fruta": fruta, "calidad": calidad, "archivo": nombre,
                })
    return pd.DataFrame(filas)


def quitar_aumentadas(df):
    df = df.copy()
    df["base"] = df["archivo"].str.replace(RE_AUMENTADA, "", regex=True).str.replace(r"\.jpe?g$", "", regex=True, case=False)
    df["es_copia"] = df["archivo"].str.contains(RE_AUMENTADA)
    df = df.sort_values(["clase", "base", "es_copia", "archivo"])
    limpio = df.drop_duplicates(subset=["clase", "base"], keep="first")
    quitadas = len(df) - len(limpio)
    return limpio.drop(columns=["base", "es_copia"]), quitadas


def asignar_sesiones(df):
    df = df.sort_values(["clase", "timestamp"]).copy()
    sesiones, contador = [], 0
    for clase, g in df.groupby("clase", sort=False):
        anterior = None
        for t in g["timestamp"]:
            if pd.isna(t) or anterior is None or pd.isna(anterior) or t - anterior > GAP_SEG:
                contador += 1
            sesiones.append(f"{clase}_s{contador:05d}")
            anterior = t
    df["sesion"] = sesiones
    return df


def dividir(df):
    rng = random.Random(SEMILLA)
    asignacion = {}
    for clase, g in df.groupby("clase"):
        tam = g.groupby("sesion").size().to_dict()
        sesiones = list(tam)
        rng.shuffle(sesiones)
        sesiones.sort(key=lambda s: -tam[s])          
        total = sum(tam.values())
        actual = {s: 0 for s in PROPORCIONES}
        for s in sesiones:
            destino = max(PROPORCIONES, key=lambda k: PROPORCIONES[k] * total - actual[k])
            asignacion[s] = destino
            actual[destino] += tam[s]
    df = df.copy()
    df["split"] = df["sesion"].map(asignacion)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--origen", required=True, help="Carpeta que contiene 'Good Quality_Fruits' y 'Bad Quality_Fruits'")
    ap.add_argument("--salida", default="data", help="Carpeta destino de train/val/test")
    ap.add_argument("--solo-csv", action="store_true", help="Solo genera splits.csv, sin copiar imágenes")
    args = ap.parse_args()

    df = leer_dataset(args.origen)
    print(f"Imágenes originales (buena + mala calidad): {len(df)}")
    resumen = {"originales": df.groupby("clase").size()}

    df, n_aug = quitar_aumentadas(df)
    print(f"Copias aumentadas eliminadas: {n_aug}")

    df["md5"] = [md5(os.path.join(args.origen, r)) for r in df["ruta"]]
    conflictos = df.groupby("md5")["clase"].nunique()
    en_varias_clases = set(conflictos[conflictos > 1].index)
    antes = len(df)
    df = df[~df["md5"].isin(en_varias_clases)].drop_duplicates(subset="md5")
    print(f"Duplicados exactos eliminados: {antes - len(df)} "
          f"({len(en_varias_clases)} hashes aparecían en más de una clase)")

    df["timestamp"] = [fecha_captura(os.path.join(args.origen, r), a) for r, a in zip(df["ruta"], df["archivo"])]
    print(f"Imágenes con orden por número de secuencia: {(df['timestamp'] >= 1e12).sum()}")
    print(f"Imágenes sin fecha ni secuencia (sesión propia): {df['timestamp'].isna().sum()}")

    df = asignar_sesiones(df)
    df = dividir(df)
    resumen["limpias"] = df.groupby("clase").size()
    resumen["sesiones"] = df.groupby("clase")["sesion"].nunique()
    for s in PROPORCIONES:
        resumen[s] = df[df["split"] == s].groupby("clase").size()

    tabla = pd.DataFrame(resumen).fillna(0).astype(int)
    tabla.loc["TOTAL"] = tabla.sum()
    print("\nResumen por clase:\n", tabla.to_string())
    print("\nProporción final:", (df["split"].value_counts(normalize=True).round(3)).to_dict())

    assert df.groupby("sesion")["split"].nunique().max() == 1

    os.makedirs(args.salida, exist_ok=True)
    df.drop(columns=["md5"]).to_csv(os.path.join(args.salida, "splits.csv"), index=False)
    tabla.to_csv(os.path.join(args.salida, "resumen_limpieza.csv"))
    print(f"\nGuardado: {os.path.join(args.salida, 'splits.csv')}")

    if not args.solo_csv:
        for r in df.itertuples():
            destino = os.path.join(args.salida, r.split, r.clase)
            os.makedirs(destino, exist_ok=True)
            shutil.copy2(os.path.join(args.origen, r.ruta), os.path.join(destino, r.archivo))
        print(f"Imágenes copiadas a {args.salida}/{{train,val,test}}/<clase>/")


if __name__ == "__main__":
    main()
