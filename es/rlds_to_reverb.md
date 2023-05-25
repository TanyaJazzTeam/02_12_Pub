# RLDS a Reverb util en TF-Agents

[RLDS](https://github.com/google-research/rlds) to [Reverb](https://github.com/deepmind/reverb) util es una herramienta en TF Agents para leer los episodios de RLDS, transformarlos en trayectorias y enviarlos a Reverb.

### Conjunto de datos RLDS

RLDS (Conjuntos de datos de aprendizaje por refuerzo) es un ecosistema de herramientas para almacenar, recuperar y manipular datos episódicos en el contexto de la toma de decisiones secuencial, incluido el aprendizaje por refuerzo (RL), el aprendizaje a partir de demostraciones, el RL sin conexión o el aprendizaje por imitación.

Cada paso tiene los siguientes campos (y, a veces, campos adicionales para los metadatos del paso). Como ejemplo, usamos las especificaciones del conjunto de datos D4RL [half-cheetah/v0-expert](https://www.tensorflow.org/datasets/catalog/d4rl_mujoco_halfcheetah#d4rl_mujoco_halfcheetahv0-expert_default_config)

- **'acción'** : `TensorSpec(shape = (6,), dtype = tf.float32, name = None)`

- **'descuento'** : `TensorSpec(shape = (), dtype = tf.float32, name = None)`

- **'es_primero'** : `TensorSpec(shape = (), dtype = tf.bool, name = None)`

- **'is_last'** : `TensorSpec(shape = (), dtype = tf.bool, name = None)`

- **'is_terminal'** : `TensorSpec(shape = (), dtype = tf.bool, name = None)`

- **'observación'** : `TensorSpec(shape = (17,), dtype = tf.float32, name = None)`

- **'recompensa'** : `TensorSpec(shape = (), dtype = tf.float32, name = None)}, TensorShape([]))`

## API de RLDS a TF-Agents utils

### Crear especificación de trayectoria a partir de un conjunto de datos

Crea especificaciones de datos para inicializar el servidor de Reverb y el búfer de reproducción de Reverb.

```
def create_trajectory_data_spec(rlds_data: tf.data.Dataset) -> trajectory.Trajectory:
```

Crea la especificación de datos para el conjunto de datos de trayectoria correspondiente que se puede crear utilizando los `rlds_data` proporcionados como entrada. Esta especificación de datos es necesaria para inicializar un servidor de reverberación y un búfer de reproducción de reverberación.

**Argumentos** :

- `rlds_data` : un conjunto de datos de RLDS es un `tf.data.Dataset` de episodios de RLDS, donde cada episodio contiene un `tf.data.Dataset` de pasos de RLDS y, opcionalmente, metadatos de episodios. Un paso RLDS es un diccionario de tensores que contiene `is_first` , `is_last` , `observation` , `action` , `reward` , `is_terminal` y `discount` (y, a veces, metadatos de pasos).

**Devoluciones** :

- Una especificación de trayectoria que se puede usar para crear un conjunto de datos de trayectoria con los `rlds_data` proporcionados como entrada.

**aumenta** :

- `ValueError` : si no existen pasos RLDS en `rlds_data` .

### Convierta datos RLDS en trayectorias de agentes TF

Convierte los datos RLDS en un conjunto de datos de trayectorias. Actualmente, solo admitimos la conversión a una trayectoria de dos pasos.

```
def convert_rlds_to_trajectories(rlds_data: tf.data.Dataset,
    policy_info_fn: _PolicyFnType = None) -> tf.data.Dataset:
```

Convierte los `rlds_data` proporcionados en un conjunto de datos de trayectorias de agentes TF aplanándolos y convirtiéndolos en lotes y luego en tuplas de pares superpuestos de pasos RLDS adyacentes.

Los datos de RLDS se completan al final con un paso de tipo `first` para garantizar que la trayectoria creada con el último paso del último episodio tenga un tipo de paso siguiente válido.

**Argumentos** :

- `rlds_data` : un conjunto de datos de RLDS es `tf.data.Dataset` de episodios de RLDS, donde cada episodio contiene un `tf.data.Dataset` de pasos de RLDS (y, opcionalmente, metadatos de episodios). Un paso RLDS es un diccionario de tensores que contiene `is_first` , `is_last` , `observation` , `action` , `reward` , `is_terminal` y `discount` (y, opcionalmente, metadatos de paso).
- `policy_info_fn` : una función opcional para crear algunos policy.info que se utilizarán al generar trayectorias de TF-Agents.

**Devoluciones** :

- Un conjunto de datos de tipo `tf.data.Dataset` , cuyos elementos son trayectorias de agentes TF correspondientes a los pasos RLDS proporcionados en `rlds_data` .

**aumenta** :

- `ValueError` : si no existen pasos RLDS en `rlds_data` .

- `InvalidArgumentError` : si el conjunto de datos RLDS proporcionado tiene episodios que:

    - Termina incorrectamente, es decir, no termina en el último paso.
    - Terminar incorrectamente, es decir, un paso terminal no es el último paso.
    - Comenzar incorrectamente, es decir, un último paso no es seguido por el primer paso. Tenga en cuenta que el último paso del último episodio se ocupa de la función y el usuario no necesita asegurarse de que el último paso del último episodio sea seguido por un primer paso.

### Transfiere datos RLDS a Reverb

Envía los datos de RLDS al servidor de Reverb como trayectorias de TF Agents. Se debe crear una instancia del observador de reverberación antes de llamar a la interfaz y se debe proporcionar como un parámetro.

```
def push_rlds_to_reverb(rlds_data: tf.data.Dataset, reverb_observer: Union[
    reverb_utils.ReverbAddEpisodeObserver,
    reverb_utils.ReverbAddTrajectoryObserver],
    policy_info_fn: _PolicyFnType = None) -> int:
```

Envía los `rlds_data` proporcionados al servidor de Reverb mediante `reverb_observer` después de convertirlos en trayectorias de TF Agents.

Tenga en cuenta que la especificación de datos utilizada para inicializar el búfer de reproducción y el servidor de reverberación para crear el `reverb_observer` debe coincidir con la especificación de datos para `rlds_data` .

**Argumentos** :

- `rlds_data` : un conjunto de datos RLDS es un `tf.data.Dataset` de episodios RLDS, donde cada episodio contiene un `tf.data.Dataset` de pasos RLDS (y, opcionalmente, metadatos de episodios). Un paso RLDS es un diccionario de tensores que contiene `is_first` , `is_last` , `observation` , `action` , `reward` , `is_terminal` y `discount` (y, opcionalmente, metadatos de paso).
- `reverb_observer` : un observador de Reverb para escribir datos de trayectorias en Reverb.
- `policy_info_fn` : una función opcional para crear algunos policy.info que se utilizarán al generar trayectorias de TF-Agents.

**Devoluciones** :

- Un `int` que representa el número de trayectorias enviadas con éxito a RLDS.

**aumenta** :

- `ValueError` : si no existen pasos RLDS en `rlds_data` .

- `ValueError` : si la especificación de datos utilizada para inicializar el búfer de reproducción y el servidor de reverberación para crear el `reverb_observer` no coincide con la especificación de datos para el conjunto de datos de trayectoria que se puede crear utilizando `rlds_data` .

- `InvalidArgumentError` : si el conjunto de datos RLDS proporcionado tiene episodios que son:

    - Termina incorrectamente, es decir, no termina en el último paso.
    - Terminar incorrectamente, es decir, un paso terminal no es el último paso.
    - Comenzar incorrectamente, es decir, un último paso no es seguido por el primer paso. Tenga en cuenta que el último paso del último episodio se ocupa de la función y el usuario no necesita asegurarse de que el último paso del último episodio sea seguido por un primer paso.

## Cómo se asignan los pasos de RLDS a las trayectorias de los agentes de TF

La siguiente secuencia son pasos RLDS en los pasos de tiempo t, t+1 y t+2. Cada paso contiene una observación (o), una acción (a), una recompensa (r) y un descuento (d). Los elementos de un mismo paso se agrupan entre paréntesis.

```
(o_t, a_t, r_t, d_t), (o_t+1, a_t+1, r_t+1, d_t+1), (o_t+2, a_t+2, r_t+2, d_t+2)
```

En RLDS,

- `o_t` corresponde a la observación en el tiempo t

- `a_t` corresponde a la acción en el tiempo t

- `r_t` corresponde a la recompensa recibida por haber realizado la acción en la observación `o_t`

- `d_t` corresponde al descuento aplicado a la recompensa `r_t`

```
Step 1 =  o_0, a_0, r_0, d_0, is_first = true, is_last = false, is_terminal = false
```

```
Step 2 =  o_1, a_1, r_1,d_1, is_first = False, is_last = false, is_terminal = false
```

…

```
Step n =  o_t, a_t, r_t, d_t, is_first = False, is_last = false, is_terminal = false
```

```
Step n+1 =   o_t+1, a_t+1, r_t+1, d_t+1, is_first = False, is_last = true, is_terminal = false
```

Cuando `is_terminal = True` , la observación corresponde a un estado final, por lo que la recompensa, el descuento y la acción no tienen sentido. Dependiendo del entorno, la observación final también puede no tener sentido.

Si un episodio termina en un paso donde `is_terminal = False` , significa que este episodio se ha truncado. En este caso, según el entorno, la acción, la recompensa y el descuento también pueden estar vacíos.

![Paso de RLDS a la trayectoria de TF-Agents](images/rlds/rlds_step_to_trajectory.png)

### Proceso de conversión

#### Aplanar el conjunto de datos

El conjunto de datos de RLDS es un conjunto de datos de episodios que, a su vez, son conjuntos de datos de pasos de RLDS. Primero se aplana a un conjunto de datos de pasos.

![Aplanar RLDS](images/rlds/flatten_rlds.png)

#### Crear pares superpuestos de pasos adyacentes

Luego, el conjunto de datos RLDS aplanado se procesa por lotes y se convierte en un conjunto de datos de pares superpuestos de pasos RLDS adyacentes.

![RLDS a pares superpuestos](images/rlds/rlds_to_pairs.png)

#### Convertir a trayectorias de TF-Agents

Luego, el conjunto de datos se convierte en trayectorias de TF-Agents.

![Pares de RLDS con trayectorias de TF-Agents](images/rlds/pairs_to_trajectories.png)
