import { clearLinks, NeuralNetwork, TrainConfig } from "./nn_organized";
import fs from "fs";
import { Descriptor } from "./descriptors/descriptor";

type BechnmarkConfig = {
    v_set:({inputs: number[], desired_outputs: number[]}[]),
    train_config:TrainConfig,
    get_prediction?:(output:number) => number,
    get_prediction_from_array?:(output:number[]) => number[],
    on_run_end?:(run:number, last_error:number) => void,
    rede:NeuralNetwork,
    bechnmark_name:string,
    runs?: number,
    descriptor?: Descriptor,
    dontSave?: boolean,
}
export function benchmark({
    descriptor,
    v_set,
    train_config,
    get_prediction = (output:number) => output,
    get_prediction_from_array = (output:number[]) => output,
    on_run_end,
    rede,
    bechnmark_name,
    runs = 10,
    dontSave = false,
}:BechnmarkConfig) {
    const start = Date.now()
    const resultados:any = {
        "epochs": train_config.epochs,
        "iteracoes": train_config.iteracoes,
        "taxa_aprendizado": train_config.taxa_aprendizado,
        "runs": runs,
        "tempo": 0,
        "momentum": train_config.momentum,
        "runs_data": [],
        "runs_avg": {},
        "layers": rede.layers.length,
        "neurons_per_layer": rede.layers.map(layer => layer.neurons.length),
        "bias_type": rede.layers[0].config.bias ? "as-input-neuron" : "no-bias",
        "activation_functions": rede.layers.map(layer => layer.config.activation_function || 'sigmoid')
    }
    if (descriptor) {
        resultados.descriptor = descriptor.name
    }
    for (let i = 0; i < resultados.runs; i++) {
        console.log('Iniciando run ' + (i+1) + ' de ' + resultados.runs)
        let resultado = {
            last_error: 0,
            precisao: 0,
            positivos: 0,
            negativos: 0
        }
        clearLinks()
        rede.createWeights()
        const train_result = rede.train(train_config)
        resultado.last_error = train_result.last_error
        
        rede.test(v_set, (err, output, desired) => {
            const prediction = get_prediction_from_array(output.map(get_prediction)).toString()
            prediction === desired.toString() ? resultado.positivos++ : resultado.negativos++
        })

        resultado.precisao = resultado.positivos / (resultado.positivos + resultado.negativos)
        resultados.runs_data.push(resultado)
        
        if (on_run_end) {
            on_run_end(i, resultado.last_error)
        }
    }
    const runs_sum = resultados.runs_data.reduce((acc, cur) => {
        acc.last_error += cur.last_error
        acc.positivos += cur.positivos
        acc.negativos += cur.negativos
        acc.precisao += cur.precisao
        return acc
    }, {
        last_error: 0,
        precisao: 0,
        positivos: 0,
        negativos: 0
    })

    resultados.runs_avg = {
        last_error: runs_sum.last_error / resultados.runs,
        precisao: runs_sum.precisao / resultados.runs,
        positivos: runs_sum.positivos / resultados.runs,
        negativos: runs_sum.negativos / resultados.runs
    }
    if (dontSave) {return}

    const end = Date.now()
    // tempo em segundos
    resultados.tempo = (end - start) / 1000

    // write a file with the benchmark name
    fs.writeFileSync(`./benchmarks/${bechnmark_name}.json`, JSON.stringify(resultados, null, 2));
}