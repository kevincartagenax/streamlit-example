{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyOEEUCSfCea7zZz2Zx0pG7s",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/kevincartagenax/streamlit-example/blob/master/app\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import numpy as np\n",
        "import pandas as pd\n",
        "import tensorflow as tf\n",
        "from tensorflow import keras\n",
        "from tensorflow.keras import layers\n",
        "\n",
        "# Cargar y preprocesar los datos\n",
        "data = pd.read_csv(\"/content/database.csv\")\n",
        "# ... Preprocesar los datos ...\n",
        "\n",
        "# Dividir los datos en características (X) y etiquetas (y)\n",
        "X = data.drop(columns=['Target'])\n",
        "y = data['Target']\n",
        "\n",
        "# Dividir los datos en conjuntos de entrenamiento y prueba\n",
        "from sklearn.model_selection import train_test_split\n",
        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
        "\n",
        "# Construir el modelo\n",
        "model = keras.Sequential([\n",
        "    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),\n",
        "    layers.Dense(32, activation='relu'),\n",
        "    layers.Dense(1, activation='sigmoid')\n",
        "])\n",
        "\n",
        "model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])\n",
        "\n",
        "# Entrenar el modelo\n",
        "model.fit(X_train, y_train, epochs=10, batch_size=32)\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "SJIZJz-rG776",
        "outputId": "2b8ec0d5-4a56-449f-bc82-5caa3ea83456"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/10\n",
            "2500/2500 [==============================] - 6s 2ms/step - loss: 0.8568 - accuracy: 0.5004\n",
            "Epoch 2/10\n",
            "2500/2500 [==============================] - 4s 2ms/step - loss: 0.7551 - accuracy: 0.4993\n",
            "Epoch 3/10\n",
            "2500/2500 [==============================] - 6s 2ms/step - loss: 0.7419 - accuracy: 0.4978\n",
            "Epoch 4/10\n",
            "2500/2500 [==============================] - 4s 2ms/step - loss: 0.7252 - accuracy: 0.4981\n",
            "Epoch 5/10\n",
            "2500/2500 [==============================] - 4s 2ms/step - loss: 0.7144 - accuracy: 0.4960\n",
            "Epoch 6/10\n",
            "2500/2500 [==============================] - 6s 2ms/step - loss: 0.7042 - accuracy: 0.5010\n",
            "Epoch 7/10\n",
            "2500/2500 [==============================] - 4s 2ms/step - loss: 0.6985 - accuracy: 0.4990\n",
            "Epoch 8/10\n",
            "2500/2500 [==============================] - 5s 2ms/step - loss: 0.6950 - accuracy: 0.5015\n",
            "Epoch 9/10\n",
            "2500/2500 [==============================] - 5s 2ms/step - loss: 0.6942 - accuracy: 0.4993\n",
            "Epoch 10/10\n",
            "2500/2500 [==============================] - 4s 2ms/step - loss: 0.6934 - accuracy: 0.5026\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<keras.callbacks.History at 0x7fcd03390bb0>"
            ]
          },
          "metadata": {},
          "execution_count": 14
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "N5am2w6wX-9u"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "pip install streamlit\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "rMgqw9G5YPq2",
        "outputId": "c8b6e39c-ca2c-4894-f564-ebf620073c59"
      },
      "execution_count": 1,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Requirement already satisfied: streamlit in /usr/local/lib/python3.10/dist-packages (1.25.0)\n",
            "Requirement already satisfied: altair<6,>=4.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (4.2.2)\n",
            "Requirement already satisfied: blinker<2,>=1.0.0 in /usr/lib/python3/dist-packages (from streamlit) (1.4)\n",
            "Requirement already satisfied: cachetools<6,>=4.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (5.3.1)\n",
            "Requirement already satisfied: click<9,>=7.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (8.1.6)\n",
            "Requirement already satisfied: importlib-metadata<7,>=1.4 in /usr/lib/python3/dist-packages (from streamlit) (4.6.4)\n",
            "Requirement already satisfied: numpy<2,>=1.19.3 in /usr/local/lib/python3.10/dist-packages (from streamlit) (1.22.4)\n",
            "Requirement already satisfied: packaging<24,>=16.8 in /usr/local/lib/python3.10/dist-packages (from streamlit) (23.1)\n",
            "Requirement already satisfied: pandas<3,>=1.3.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (1.5.3)\n",
            "Requirement already satisfied: pillow<10,>=7.1.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (8.4.0)\n",
            "Requirement already satisfied: protobuf<5,>=3.20 in /usr/local/lib/python3.10/dist-packages (from streamlit) (3.20.3)\n",
            "Requirement already satisfied: pyarrow>=6.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (9.0.0)\n",
            "Requirement already satisfied: pympler<2,>=0.9 in /usr/local/lib/python3.10/dist-packages (from streamlit) (1.0.1)\n",
            "Requirement already satisfied: python-dateutil<3,>=2.7.3 in /usr/local/lib/python3.10/dist-packages (from streamlit) (2.8.2)\n",
            "Requirement already satisfied: requests<3,>=2.18 in /usr/local/lib/python3.10/dist-packages (from streamlit) (2.27.1)\n",
            "Requirement already satisfied: rich<14,>=10.14.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (13.4.2)\n",
            "Requirement already satisfied: tenacity<9,>=8.1.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (8.2.2)\n",
            "Requirement already satisfied: toml<2,>=0.10.1 in /usr/local/lib/python3.10/dist-packages (from streamlit) (0.10.2)\n",
            "Requirement already satisfied: typing-extensions<5,>=4.1.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (4.7.1)\n",
            "Requirement already satisfied: tzlocal<5,>=1.1 in /usr/local/lib/python3.10/dist-packages (from streamlit) (4.3.1)\n",
            "Requirement already satisfied: validators<1,>=0.2 in /usr/local/lib/python3.10/dist-packages (from streamlit) (0.20.0)\n",
            "Requirement already satisfied: gitpython!=3.1.19,<4,>=3.0.7 in /usr/local/lib/python3.10/dist-packages (from streamlit) (3.1.32)\n",
            "Requirement already satisfied: pydeck<1,>=0.8 in /usr/local/lib/python3.10/dist-packages (from streamlit) (0.8.0)\n",
            "Requirement already satisfied: tornado<7,>=6.0.3 in /usr/local/lib/python3.10/dist-packages (from streamlit) (6.3.1)\n",
            "Requirement already satisfied: watchdog>=2.1.5 in /usr/local/lib/python3.10/dist-packages (from streamlit) (3.0.0)\n",
            "Requirement already satisfied: entrypoints in /usr/local/lib/python3.10/dist-packages (from altair<6,>=4.0->streamlit) (0.4)\n",
            "Requirement already satisfied: jinja2 in /usr/local/lib/python3.10/dist-packages (from altair<6,>=4.0->streamlit) (3.1.2)\n",
            "Requirement already satisfied: jsonschema>=3.0 in /usr/local/lib/python3.10/dist-packages (from altair<6,>=4.0->streamlit) (4.3.3)\n",
            "Requirement already satisfied: toolz in /usr/local/lib/python3.10/dist-packages (from altair<6,>=4.0->streamlit) (0.12.0)\n",
            "Requirement already satisfied: gitdb<5,>=4.0.1 in /usr/local/lib/python3.10/dist-packages (from gitpython!=3.1.19,<4,>=3.0.7->streamlit) (4.0.10)\n",
            "Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.10/dist-packages (from pandas<3,>=1.3.0->streamlit) (2022.7.1)\n",
            "Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.10/dist-packages (from python-dateutil<3,>=2.7.3->streamlit) (1.16.0)\n",
            "Requirement already satisfied: urllib3<1.27,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.18->streamlit) (1.26.16)\n",
            "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.18->streamlit) (2023.5.7)\n",
            "Requirement already satisfied: charset-normalizer~=2.0.0 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.18->streamlit) (2.0.12)\n",
            "Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.18->streamlit) (3.4)\n",
            "Requirement already satisfied: markdown-it-py>=2.2.0 in /usr/local/lib/python3.10/dist-packages (from rich<14,>=10.14.0->streamlit) (3.0.0)\n",
            "Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /usr/local/lib/python3.10/dist-packages (from rich<14,>=10.14.0->streamlit) (2.14.0)\n",
            "Requirement already satisfied: pytz-deprecation-shim in /usr/local/lib/python3.10/dist-packages (from tzlocal<5,>=1.1->streamlit) (0.1.0.post0)\n",
            "Requirement already satisfied: decorator>=3.4.0 in /usr/local/lib/python3.10/dist-packages (from validators<1,>=0.2->streamlit) (4.4.2)\n",
            "Requirement already satisfied: smmap<6,>=3.0.1 in /usr/local/lib/python3.10/dist-packages (from gitdb<5,>=4.0.1->gitpython!=3.1.19,<4,>=3.0.7->streamlit) (5.0.0)\n",
            "Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.10/dist-packages (from jinja2->altair<6,>=4.0->streamlit) (2.1.3)\n",
            "Requirement already satisfied: attrs>=17.4.0 in /usr/local/lib/python3.10/dist-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (23.1.0)\n",
            "Requirement already satisfied: pyrsistent!=0.17.0,!=0.17.1,!=0.17.2,>=0.14.0 in /usr/local/lib/python3.10/dist-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (0.19.3)\n",
            "Requirement already satisfied: mdurl~=0.1 in /usr/local/lib/python3.10/dist-packages (from markdown-it-py>=2.2.0->rich<14,>=10.14.0->streamlit) (0.1.2)\n",
            "Requirement already satisfied: tzdata in /usr/local/lib/python3.10/dist-packages (from pytz-deprecation-shim->tzlocal<5,>=1.1->streamlit) (2023.3)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!pip install pyngrok\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "MUQlgZGVZNHu",
        "outputId": "82659541-7706-4a78-f0dd-d524b5c40855"
      },
      "execution_count": 21,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Collecting pyngrok\n",
            "  Downloading pyngrok-6.0.0.tar.gz (681 kB)\n",
            "\u001b[?25l     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m0.0/681.2 kB\u001b[0m \u001b[31m?\u001b[0m eta \u001b[36m-:--:--\u001b[0m\r\u001b[2K     \u001b[91m━━━\u001b[0m\u001b[91m╸\u001b[0m\u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m61.4/681.2 kB\u001b[0m \u001b[31m1.7 MB/s\u001b[0m eta \u001b[36m0:00:01\u001b[0m\r\u001b[2K     \u001b[91m━━━━━━━━━━━━━━━━━\u001b[0m\u001b[91m╸\u001b[0m\u001b[90m━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m307.2/681.2 kB\u001b[0m \u001b[31m4.6 MB/s\u001b[0m eta \u001b[36m0:00:01\u001b[0m\r\u001b[2K     \u001b[91m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[91m╸\u001b[0m\u001b[90m━━━━\u001b[0m \u001b[32m604.2/681.2 kB\u001b[0m \u001b[31m6.0 MB/s\u001b[0m eta \u001b[36m0:00:01\u001b[0m\r\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m681.2/681.2 kB\u001b[0m \u001b[31m5.6 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25h  Preparing metadata (setup.py) ... \u001b[?25l\u001b[?25hdone\n",
            "Requirement already satisfied: PyYAML in /usr/local/lib/python3.10/dist-packages (from pyngrok) (6.0.1)\n",
            "Building wheels for collected packages: pyngrok\n",
            "  Building wheel for pyngrok (setup.py) ... \u001b[?25l\u001b[?25hdone\n",
            "  Created wheel for pyngrok: filename=pyngrok-6.0.0-py3-none-any.whl size=19867 sha256=8c4fcbefb4c157422f80579e8ab824394fcda6ae5883d19f5243632052ab479c\n",
            "  Stored in directory: /root/.cache/pip/wheels/5c/42/78/0c3d438d7f5730451a25f7ac6cbf4391759d22a67576ed7c2c\n",
            "Successfully built pyngrok\n",
            "Installing collected packages: pyngrok\n",
            "Successfully installed pyngrok-6.0.0\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "\n",
        "!ngrok authtoken 2T0OktLAosnqUL8ITHm4mHxvRwR_tkjaGrMESpWwbiq8UDp7\n",
        "\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "mEwDki81cQp9",
        "outputId": "a3fb60f7-f8b6-4846-be48-137736ca5d37"
      },
      "execution_count": 2,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Authtoken saved to configuration file: /root/.ngrok2/ngrok.yml\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import pandas as pd\n",
        "import numpy as np\n",
        "import tensorflow as tf\n",
        "from tensorflow import keras\n",
        "from tensorflow.keras import layers\n",
        "import matplotlib.pyplot as plt\n",
        "import subprocess\n",
        "import re\n",
        "\n",
        "# Cargar y preprocesar los datos desde tu base de datos\n",
        "data = pd.read_csv(\"/content/database.csv\")\n",
        "# ... Preprocesar los datos ...\n",
        "\n",
        "# Dividir los datos en características (X) y etiquetas (y)\n",
        "X = data.drop(columns=['Target'])\n",
        "y = data['Target']\n",
        "\n",
        "# Dividir los datos en conjuntos de entrenamiento y prueba\n",
        "from sklearn.model_selection import train_test_split\n",
        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n"
      ],
      "metadata": {
        "id": "Pzc6C-eefYFM"
      },
      "execution_count": 3,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Construir el modelo\n",
        "model = keras.Sequential([\n",
        "    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),\n",
        "    layers.Dense(32, activation='relu'),\n",
        "    layers.Dense(1, activation='sigmoid')\n",
        "])\n",
        "\n",
        "model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])\n",
        "\n",
        "# Entrenar el modelo\n",
        "model.fit(X_train, y_train, epochs=10, batch_size=32)\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "G-R4qNq4feFM",
        "outputId": "3f4f67d0-5bc8-48b0-cdb4-053726ec9509"
      },
      "execution_count": 4,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/10\n",
            "2500/2500 [==============================] - 7s 2ms/step - loss: 0.8938 - accuracy: 0.4999\n",
            "Epoch 2/10\n",
            "2500/2500 [==============================] - 6s 2ms/step - loss: 0.7566 - accuracy: 0.5006\n",
            "Epoch 3/10\n",
            "2500/2500 [==============================] - 5s 2ms/step - loss: 0.7317 - accuracy: 0.4998\n",
            "Epoch 4/10\n",
            "2500/2500 [==============================] - 4s 2ms/step - loss: 0.7200 - accuracy: 0.4990\n",
            "Epoch 5/10\n",
            "2500/2500 [==============================] - 6s 2ms/step - loss: 0.7112 - accuracy: 0.5005\n",
            "Epoch 6/10\n",
            "2500/2500 [==============================] - 4s 2ms/step - loss: 0.7041 - accuracy: 0.5022\n",
            "Epoch 7/10\n",
            "2500/2500 [==============================] - 4s 2ms/step - loss: 0.6983 - accuracy: 0.4987\n",
            "Epoch 8/10\n",
            "2500/2500 [==============================] - 6s 2ms/step - loss: 0.6953 - accuracy: 0.4974\n",
            "Epoch 9/10\n",
            "2500/2500 [==============================] - 4s 2ms/step - loss: 0.6940 - accuracy: 0.4965\n",
            "Epoch 10/10\n",
            "2500/2500 [==============================] - 4s 2ms/step - loss: 0.6933 - accuracy: 0.5005\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<keras.callbacks.History at 0x7960402faf20>"
            ]
          },
          "metadata": {},
          "execution_count": 4
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Función para hacer predicciones\n",
        "def predict_anemia(data):\n",
        "    prediction = model.predict(data)\n",
        "    return prediction\n"
      ],
      "metadata": {
        "id": "NTvb6N_wftzU"
      },
      "execution_count": 5,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Interfaz de la aplicación Streamlit\n",
        "def main():\n",
        "    import streamlit as st  # Importar streamlit aquí\n",
        "\n",
        "    st.title(\"Detector de Anemia\")\n",
        "\n",
        "    st.write(\"Ingrese los datos del paciente:\")\n",
        "    age = st.number_input(\"Edad\", min_value=18, max_value=100, value=50)\n",
        "    hemoglobin = st.number_input(\"Hemoglobina\", min_value=5, max_value=20, value=12)\n",
        "    ferritin = st.number_input(\"Ferritina\", min_value=0, max_value=500, value=100)\n",
        "    iron = st.number_input(\"Hierro\", min_value=0, max_value=300, value=100)\n",
        "    transferrin = st.number_input(\"Transferrina\", min_value=100, max_value=500, value=300)\n",
        "    saturation = st.number_input(\"Saturación\", min_value=0, max_value=100, value=50)\n",
        "\n",
        "    # Hacer la predicción con los datos ingresados\n",
        "    data_to_predict = pd.DataFrame({\n",
        "        'Age': [age],\n",
        "        'Hemoglobin': [hemoglobin],\n",
        "        'Ferritin': [ferritin],\n",
        "        'Iron': [iron],\n",
        "        'Transferrin': [transferrin],\n",
        "        'Saturation': [saturation]\n",
        "    })\n",
        "    prediction = predict_anemia(data_to_predict)\n",
        "\n",
        "    st.write(\"Resultado de la predicción:\")\n",
        "    if prediction[0][0] >= 0.5:\n",
        "        st.write(\"El paciente podría tener anemia.\")\n",
        "    else:\n",
        "        st.write(\"El paciente probablemente no tiene anemia.\")\n",
        "\n",
        "    # Mostrar el histograma de edades\n",
        "    st.write(\"Histograma de edades:\")\n",
        "    st.bar_chart(data['Age'])\n",
        "\n",
        "# Ejecutar ngrok para obtener el URL público de la aplicación Streamlit\n",
        "command = \"ngrok authtoken 2T0OktLAosnqUL8ITHm4mHxvRwR_tkjaGrMESpWwbiq8UDp7 && streamlit run --server.port 8501 app.py\"\n",
        "process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)\n",
        "ngrok_url = None\n",
        "while True:\n",
        "    output = process.stdout.readline().decode().strip()\n",
        "    if output == \"\":\n",
        "        break\n",
        "    match = re.match(r\"NgrokTunnel: .* -> (http.*)\", output)\n",
        "    if match:\n",
        "        ngrok_url = match.group(1)\n",
        "        break\n",
        "\n",
        "if ngrok_url:\n",
        "    print(\"  https://3b40-2800-4b0-8431-b444-c5b7-f0d-419a-3c0d.ngrok-free.app -> http://localhost:80\", ngrok_url)\n",
        "else:\n",
        "    print(\"Error al obtener el URL de ngrok.\")\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "nKk43oMKfxpU",
        "outputId": "ea96eb9e-0c86-4233-ec11-df99435cd7a4"
      },
      "execution_count": 16,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Error al obtener el URL de ngrok.\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!pip show pyngrok\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "TO30QVB-g270",
        "outputId": "47533acd-9966-4dcc-ceed-ed8f3b814703"
      },
      "execution_count": 7,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Name: pyngrok\n",
            "Version: 5.0.2\n",
            "Summary: A Python wrapper for Ngrok.\n",
            "Home-page: https://github.com/alexdlaird/pyngrok\n",
            "Author: Alex Laird\n",
            "Author-email: contact@alexlaird.com\n",
            "License: MIT\n",
            "Location: /usr/local/lib/python3.10/dist-packages\n",
            "Requires: PyYAML\n",
            "Required-by: \n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!cat /root/.ngrok2/ngrok.yml\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "eIqrSSa5gGzF",
        "outputId": "84558bd3-e834-4c13-c7c6-3d74e52ff39b"
      },
      "execution_count": 8,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "region: us\n",
            "version: '2'\n",
            "authtoken: 2T0OktLAosnqUL8ITHm4mHxvRwR_tkjaGrMESpWwbiq8UDp7\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "command = \"streamlit run --server.port 8501 app.py\"\n"
      ],
      "metadata": {
        "id": "z2pxa4Rrgs5k"
      },
      "execution_count": 9,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "!ls\n",
        "\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "5SIgpR0_jk-L",
        "outputId": "9a4d4825-7c9b-440e-fe41-236dc25aec06"
      },
      "execution_count": 12,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "database.csv  sample_data\n"
          ]
        }
      ]
    }
  ]
}