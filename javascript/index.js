function sometin(){
    debugger
        // Seleccionar el contenedor principal
        const container = document.querySelector('#container');

        // Crear y agregar nuevos divs con párrafos
        for (let i = 1; i <= 3; i++) {
            // Crear un nuevo div
            const newDiv = document.createElement('div');
            newDiv.className = `div-${i}`;
            newDiv.style.border = "1px solid black";
            newDiv.style.margin = "10px";
            newDiv.style.padding = "10px";
    
            // Crear un nuevo párrafo dentro del div
            const newParagraph = document.createElement('p');
            newParagraph.textContent = `Este es el párrafo ${i} dentro del div-${i}`;
    
            // Agregar el párrafo al div
            newDiv.appendChild(newParagraph);
    
            // Agregar el div al contenedor principal
            container.appendChild(newDiv);
        }
    
        console.log('Nuevos divs y párrafos añadidos.');
}

sometin();

document.getElementById("paco").addEventListener("click", function(){
    alert("hola");
    
})