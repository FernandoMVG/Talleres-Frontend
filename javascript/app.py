import os
import json
import torch
from typing import List
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    prepare_model_for_kbit_training,
    PeftModel
)
from tqdm import tqdm

class TranscriptionUnifier:
    def __init__(self, model_path="mistralai/Mistral-7B-Instruct-v0.2"):
        """Inicializa el unificador de transcripciones con un modelo local"""
        print(f"Cargando modelo desde {model_path}...")
        
        # Configuración para cargar el modelo en formato de menor precisión para ahorrar memoria
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        
        # Cargar tokenizer y modelo
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Configurar tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        
        print("Modelo cargado con éxito")

    def unify_transcripts(self, fragments: List[str]) -> str:
        """Unifica fragmentos de transcripción usando el modelo local"""
        
        # Construir el prompt
        prompt = f"""<s>[INST] Unifica estos fragmentos de transcripción en un texto coherente:

{chr(10).join([f"Fragmento {i+1}: {frag}" for i, frag in enumerate(fragments)])}

Elimina repeticiones y organiza todo en un documento fluido. [/INST]
"""

        # Configurar generación
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # Generar respuesta
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,
                do_sample=True
            )
        
        # Decodificar respuesta
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extraer la parte después del prompt
        response = response.split("[/INST]")[-1].strip()
        
        return response


def prepare_dataset(data_file):
    """Prepara un dataset de ejemplos para fine-tuning"""
    
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    formatted_data = []
    
    for example in data:
        fragments = example["fragments"]
        unified = example["unified"]
        
        prompt = f"""<s>[INST] Unifica estos fragmentos de transcripción en un texto coherente:

{chr(10).join([f"Fragmento {i+1}: {frag}" for i, frag in enumerate(fragments)])}

Elimina repeticiones y organiza todo en un documento fluido. [/INST]

{unified}</s>"""
        
        formatted_data.append({"text": prompt})
    
    return Dataset.from_list(formatted_data)


def fine_tune_mistral(dataset_path, output_dir, model_name="mistralai/Mistral-7B-Instruct-v0.2"):
    """Fine-tunea un modelo Mistral para unificación de transcripciones"""
    
    print("Preparando dataset...")
    dataset = prepare_dataset(dataset_path)
    
    print(f"Cargando modelo base {model_name}...")
    
    # Configuración para cargar el modelo en formato de menor precisión
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    # Cargar modelo y tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Preparar modelo para el entrenamiento
    model = prepare_model_for_kbit_training(model)
    
    # Configurar LoRA para fine-tuning eficiente
    lora_config = LoraConfig(
        r=16,  # Rango de la matriz de adaptación
        lora_alpha=32,  # Parámetro de escala
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Capas a adaptar
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Aplicar configuración LoRA al modelo
    model = get_peft_model(model, lora_config)
    
    print(f"Parámetros entrenables: {model.print_trainable_parameters()}")
    
    # Tokenizar dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=2048,
            padding="max_length"
        )
    
    print("Tokenizando dataset...")
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # Colator de datos
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Configurar entrenamiento
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_steps=100,
        warmup_steps=50,
        lr_scheduler_type="cosine",
        save_total_limit=3,
        push_to_hub=False,
    )
    
    # Iniciar entrenamiento
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    
    print("Iniciando fine-tuning...")
    trainer.train()
    
    # Guardar modelo y tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"Modelo fine-tuned guardado en {output_dir}")


# Ejemplo de uso
if __name__ == "__main__":
    # Crear dataset de ejemplo para fine-tuning
    example_dataset = [
        {
            "fragments": [
                "Bueno, entonces como les decía, el proyecto tiene tres fases principales.",
                "La primera fase del proyecto consiste en recolectar los datos y analizarlos en detalle.",
                "Después de analizar los datos pasaremos a la segunda fase del proyecto.",
                "En la segunda fase implementaremos el modelo basado en los análisis previos.",
                "Y finalmente, en la tercera fase haremos pruebas y ajustes para optimizar."
            ],
            "unified": "El proyecto tiene tres fases principales. La primera fase consiste en recolectar y analizar los datos en detalle. En la segunda fase, implementaremos el modelo basado en los análisis previos. Finalmente, en la tercera fase realizaremos pruebas y ajustes para optimizar el sistema."
        },
        # Agrega más ejemplos para mejor entrenamiento
    ]
    
    # Guardar dataset de ejemplo
    os.makedirs("data", exist_ok=True)
    with open("data/dataset_transcripciones.json", "w", encoding="utf-8") as f:
        json.dump(example_dataset, f, ensure_ascii=False, indent=2)
    
    # Realizar fine-tuning (descomentar para ejecutar)
    """
    fine_tune_mistral(
        dataset_path="data/dataset_transcripciones.json",
        output_dir="./mistral-transcription-unifier"
    )
    """
    
    # Usar el modelo (sin fine-tuning o con fine-tuning)
    # Para usar modelo fine-tuned, cambia el path a "./mistral-transcription-unifier"
    unifier = TranscriptionUnifier()
    
    # Ejemplo de fragmentos para unificar
    fragments = [
        "Bueno, entonces como les decía, el proyecto tiene tres fases principales.",
        "La primera fase del proyecto consiste en recolectar los datos y analizarlos.",
        "Después de analizar los datos, pasaremos a la segunda fase.",
        "En la segunda fase implementaremos el modelo basado en los análisis previos.",
        "Y finalmente, en la tercera fase haremos pruebas y ajustes."
    ]
    
    print("Unificando transcripción...")
    unified_transcript = unifier.unify_transcripts(fragments)
    print("\nTRANSCRIPCIÓN UNIFICADA:")
    print(unified_transcript)