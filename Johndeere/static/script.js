// static/script.js

function fazerPrevisao() {
    var tipo_peca = document.getElementById("tipo_peca").value;
    var quantidade_manutencao = document.getElementById("quantidade_manutencao").value;
    var tempo_funcionamento = document.getElementById("tempo_funcionamento").value;
    var modelo_trator = document.getElementById("modelo_trator").value;

    var data = {
        "tipo_peca": tipo_peca,
        "quantidade_manutencao": parseInt(quantidade_manutencao),
        "tempo_funcionamento": parseInt(tempo_funcionamento),
        "modelo_trator": modelo_trator
    };

    fetch("/fazer_previsao", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById("vida_util_prevista").textContent = data.vida_util_prevista;
        document.getElementById("tempo_restante").textContent = data.tempo_restante;
    })
    .catch(error => console.error(error));
}
