# Diretriz Curada — Detecção Precoce do Câncer de Mama

## 1. Identificação da fonte

### 1.1 Fonte principal

Este documento curado foi elaborado a partir da seguinte referência oficial:

**Instituto Nacional de Câncer José Alencar Gomes da Silva (INCA). Ministério da Saúde.  
Diretrizes para a Detecção Precoce do Câncer de Mama no Brasil. Rio de Janeiro: INCA, 2015.**

A publicação apresenta diretrizes nacionais para apoiar a detecção precoce do câncer de mama no Brasil. O documento foi elaborado pelo INCA/Ministério da Saúde com base em revisão de evidências científicas, avaliação de benefícios e danos das intervenções e graduação das recomendações pelo método GRADE.

### 1.2 Finalidade deste arquivo curado

Este arquivo é uma versão curada, resumida e estruturada da diretriz oficial, preparada para uso como fonte de conhecimento no projeto acadêmico **FemCare AI**.

Sua finalidade é apoiar respostas de RAG e documentação técnica sobre:

- detecção precoce do câncer de mama;
- rastreamento em população de risco padrão;
- diagnóstico precoce;
- mamografia;
- autoexame das mamas;
- conscientização corporal;
- sinais e sintomas suspeitos;
- encaminhamento para avaliação profissional.

Este arquivo não substitui a diretriz oficial completa. Ele reorganiza pontos essenciais da referência para facilitar recuperação semântica, rastreabilidade e uso seguro em um MVP acadêmico.

### 1.3 Natureza da curadoria

Este documento contém:

- paráfrases fiéis da diretriz;
- sínteses estruturadas de recomendações;
- reorganização do conteúdo em formato Markdown;
- linguagem simplificada para uso em RAG;
- preservação da fonte e do contexto original.

Este documento não deve adicionar recomendações clínicas próprias que não estejam fundamentadas na fonte.

As regras específicas de funcionamento do FemCare AI, como classificação de risco, retorno JSON da tool, Safety Validator, logs e integração com o modelo da Fase 2, devem ser documentadas separadamente em arquivo técnico, por exemplo:

`docs/breast_cancer_operational_rules.md`

### 1.4 Uso recomendado no projeto

Este arquivo pode ser usado como:

- fonte curada para RAG;
- referência documental do fluxo Breast Cancer;
- base textual para respostas sobre rastreamento e diagnóstico precoce;
- fonte citada pelo assistente;
- referência para relatório técnico.

Uso recomendado no retorno de fontes:

~~~json
{
  "sources": [
    "data/protocols/breast_cancer_screening.md",
    "Diretrizes para a Detecção Precoce do Câncer de Mama no Brasil - INCA/Ministério da Saúde, 2015"
  ]
}
~~~

### 1.5 Nota de segurança

Este arquivo trata de informações de saúde e deve ser usado apenas como apoio informacional.

O conteúdo não deve ser usado para:

- diagnóstico definitivo;
- prescrição de medicamentos;
- definição de tratamento;
- substituição de consulta médica;
- interpretação definitiva de exames;
- descarte de câncer;
- confirmação de câncer.

Quando houver sinais ou sintomas suspeitos, a orientação segura é buscar avaliação profissional.

---

## 2. Escopo da diretriz

### 2.1 Escopo geral

A diretriz do INCA/Ministério da Saúde aborda a **detecção precoce do câncer de mama no Brasil**.

O foco principal é avaliar estratégias de detecção precoce, considerando possíveis benefícios e danos das intervenções disponíveis.

A diretriz trata principalmente de duas estratégias:

1. **Rastreamento**
2. **Diagnóstico precoce**

Essas estratégias têm finalidades diferentes e devem ser compreendidas separadamente.

### 2.2 Rastreamento em população de risco padrão

A diretriz avalia estratégias de rastreamento para mulheres de **risco padrão**, também chamadas de população de risco populacional.

No contexto da diretriz, risco padrão se refere à população geral, sem classificação específica de alto risco para desenvolvimento de câncer de mama.

A diretriz avalia as seguintes tecnologias ou ações de rastreamento:

- mamografia;
- autoexame das mamas;
- exame clínico das mamas;
- ressonância nuclear magnética;
- ultrassonografia mamária;
- termografia;
- tomossíntese mamária.

Para uso no FemCare AI, a seção mais relevante para rastreamento é a recomendação sobre **mamografia em mulheres de risco padrão**.

### 2.3 Diagnóstico precoce

A diretriz também aborda estratégias de diagnóstico precoce, especialmente para pessoas que já apresentam sinais ou sintomas suspeitos.

As ações de diagnóstico precoce avaliadas incluem:

- estratégias de conscientização;
- identificação de sinais e sintomas suspeitos na atenção primária;
- organização da confirmação diagnóstica em serviço de referência.

Para uso no FemCare AI, essa parte é essencial porque o assistente deve diferenciar uma pergunta preventiva de uma situação sintomática.

### 2.4 O que a diretriz não cobre completamente

A própria diretriz informa limitações de escopo. Entre os temas que não são tratados de forma completa estão:

- rastreamento específico para população de alto risco;
- avaliação econômica ou custo-efetividade;
- tratamento oncológico;
- definição de conduta terapêutica individual;
- protocolos completos para grupos genéticos específicos;
- prescrição medicamentosa;
- manejo clínico individualizado.

Portanto, este arquivo curado também não deve ser usado para cobrir esses temas como se fossem parte da recomendação original.

### 2.5 Escopo de uso no FemCare AI

No projeto FemCare AI, esta diretriz curada deve apoiar principalmente:

- perguntas educativas sobre câncer de mama;
- orientação geral sobre rastreamento;
- explicação sobre mamografia;
- distinção entre rastreamento e diagnóstico precoce;
- identificação de sinais de alerta;
- recomendação de busca por avaliação profissional;
- citação de fonte oficial.

Este arquivo não deve ser usado isoladamente para gerar conduta médica individual.

---

## 3. Conceitos da diretriz

### 3.1 Detecção precoce

A detecção precoce é uma forma de prevenção secundária. Seu objetivo é identificar o câncer em estágios iniciais, quando há maior possibilidade de melhor prognóstico.

No câncer de mama, a detecção precoce não deve ser confundida com prevenção primária.

- **Prevenção primária**: busca reduzir ou eliminar fatores de risco antes que a doença ocorra.
- **Detecção precoce**: busca identificar a doença em fase inicial ou identificar sinais suspeitos de forma oportuna.

A diretriz organiza a detecção precoce em duas estratégias principais:

1. **Rastreamento**
2. **Diagnóstico precoce**

### 3.2 Rastreamento

Rastreamento é a realização de exames ou testes em pessoas aparentemente saudáveis, sem sinais ou sintomas suspeitos, com o objetivo de identificar possível doença em fase pré-clínica.

No caso do câncer de mama, o rastreamento é voltado para pessoas assintomáticas.

Um exame de rastreamento não é diagnóstico definitivo. Quando o rastreamento apresenta resultado alterado, são necessários exames complementares e avaliação profissional para confirmação ou exclusão diagnóstica.

A diretriz destaca que qualquer estratégia de rastreamento deve ser recomendada somente quando seus benefícios superam seus possíveis danos em determinada população.

### 3.3 Diagnóstico precoce

Diagnóstico precoce é a identificação de possível câncer em pessoas que já apresentam sinais ou sintomas suspeitos.

O objetivo é reduzir atrasos entre:

- percepção dos sinais ou sintomas;
- procura por atendimento;
- avaliação profissional;
- confirmação diagnóstica;
- início do cuidado apropriado, quando necessário.

No contexto do câncer de mama, o diagnóstico precoce depende de três elementos:

1. população alerta para sinais e sintomas suspeitos;
2. profissionais de saúde preparados para avaliar casos suspeitos;
3. serviços de saúde capazes de garantir confirmação diagnóstica oportuna e continuidade do cuidado.

### 3.4 Diferença entre rastreamento e diagnóstico precoce

A distinção entre rastreamento e diagnóstico precoce é essencial.

| Estratégia | Público principal | Objetivo | Exemplo |
|---|---|---|---|
| Rastreamento | Pessoas assintomáticas | Identificar alterações antes de sintomas | Mamografia em faixa etária recomendada |
| Diagnóstico precoce | Pessoas com sinais ou sintomas | Investigar alterações suspeitas de forma oportuna | Avaliação de nódulo, secreção sanguinolenta ou retração da pele |

No uso em um assistente de IA, essa distinção é importante porque uma pessoa com sintoma suspeito não deve receber apenas orientação genérica de rastreamento de rotina.

### 3.5 População de risco padrão

A diretriz de rastreamento com mamografia avalia principalmente mulheres de risco padrão.

Risco padrão significa que a pessoa pertence à população geral e não foi classificada como tendo risco elevado por condições específicas.

A diretriz não aprofunda recomendações para população de alto risco.

Quando houver relato de forte histórico familiar, mutação genética conhecida, câncer de mama prévio ou outras condições de alto risco, a orientação deve ser individualizada por profissional de saúde.

### 3.6 Benefícios e danos

A diretriz considera que intervenções de rastreamento podem gerar benefícios, mas também danos.

Possíveis benefícios incluem detecção em fase inicial e redução de mortalidade por câncer de mama em determinadas faixas etárias.

Possíveis danos incluem:

- resultados falso-positivos;
- ansiedade;
- exames adicionais;
- biópsias desnecessárias;
- resultados falso-negativos;
- sobrediagnóstico;
- sobretratamento;
- exposição à radiação ionizante.

Por isso, as recomendações são baseadas no balanço entre benefícios e danos, e não apenas na capacidade do exame de detectar alterações.

---

## 4. Recomendações de rastreamento com mamografia

### 4.1 Papel da mamografia

A mamografia é o principal método avaliado pela diretriz para rastreamento do câncer de mama em mulheres de risco padrão.

A diretriz reconhece que a mamografia pode reduzir mortalidade por câncer de mama em determinadas faixas etárias, mas também pode causar danos. Por isso, as recomendações variam conforme a idade.

### 4.2 Recomendações por faixa etária

A diretriz apresenta recomendações específicas para rastreamento com mamografia em mulheres de risco padrão.

| Faixa etária | Recomendação da diretriz |
|---|---|
| Menos de 50 anos | Recomenda contra o rastreamento com mamografia. |
| 50 a 59 anos | Recomenda o rastreamento com mamografia, com recomendação favorável fraca. |
| 60 a 69 anos | Recomenda o rastreamento com mamografia, com recomendação favorável fraca. |
| 70 a 74 anos | Recomenda contra o rastreamento com mamografia, com recomendação contrária fraca. |
| 75 anos ou mais | Recomenda contra o rastreamento com mamografia, com recomendação contrária forte. |

### 4.3 Periodicidade

Para as faixas etárias em que o rastreamento com mamografia é recomendado, a diretriz recomenda periodicidade **bienal**.

Ou seja, para mulheres de risco padrão dentro das faixas recomendadas, a mamografia de rastreamento deve ser considerada em intervalo aproximado de dois anos.

A diretriz também destaca que periodicidades menores, como rastreamento anual, podem aumentar danos sem evidência clara de benefício adicional proporcional.

### 4.4 Interpretação da recomendação favorável fraca

A recomendação favorável fraca significa que o rastreamento pode ser apropriado, mas a decisão deve considerar:

- balanço entre benefícios e danos;
- idade;
- contexto de saúde;
- preferências da pessoa;
- avaliação profissional;
- organização do serviço de saúde.

No caso de mulheres de 50 a 59 anos, a diretriz aponta que benefícios e danos provavelmente são semelhantes.

No caso de mulheres de 60 a 69 anos, a diretriz aponta que os benefícios provavelmente superam os danos.

### 4.5 Interpretação da recomendação contrária

A recomendação contra o rastreamento em determinadas faixas etárias não significa que uma pessoa nunca possa ser avaliada.

Ela significa que a mamografia de rotina, como rastreamento populacional em pessoas assintomáticas de risco padrão, não é recomendada para aquelas faixas.

Casos individuais podem exigir avaliação profissional, especialmente quando há:

- sinais ou sintomas suspeitos;
- histórico familiar relevante;
- risco aumentado;
- exame prévio alterado;
- orientação médica específica.

### 4.6 Rastreamento versus sintomas

As recomendações de mamografia por faixa etária são voltadas para rastreamento em pessoas assintomáticas.

Quando há sinais ou sintomas suspeitos, o caso deve ser tratado como necessidade de avaliação profissional, não apenas como rastreamento de rotina.

Exemplos de sintomas que mudam o contexto:

- nódulo na mama;
- secreção sanguinolenta pelo mamilo;
- retração da pele;
- retração ou alteração do mamilo;
- caroço na axila;
- aumento progressivo da mama;
- pele com aspecto de casca de laranja.

Nesses casos, a orientação deve ser procurar avaliação profissional.

### 4.7 Uso seguro no FemCare AI

Quando o usuário perguntar sobre mamografia de forma preventiva, o FemCare AI pode informar que, segundo a diretriz brasileira do INCA/Ministério da Saúde, o rastreamento com mamografia é recomendado principalmente para mulheres de risco padrão de 50 a 69 anos, com periodicidade bienal.

Quando o usuário relatar sintomas, o FemCare AI deve evitar responder apenas com a periodicidade de rastreamento. Deve orientar avaliação profissional.

---

## 5. Autoexame das mamas

### 5.1 Definição de autoexame das mamas

A diretriz define o autoexame das mamas como um procedimento em que a mulher observa e palpa as próprias mamas e estruturas associadas, com objetivo de detectar mudanças ou anormalidades.

Historicamente, o autoexame foi ensinado como uma técnica padronizada, com periodicidade fixa, geralmente mensal.

### 5.2 Recomendação da diretriz

A diretriz recomenda contra o ensino do autoexame das mamas como método de rastreamento do câncer de mama.

A recomendação é contrária fraca, baseada na avaliação de que os possíveis danos provavelmente superam os possíveis benefícios.

### 5.3 Razões apresentadas na diretriz

A diretriz relata que grandes estudos e revisões sistemáticas não comprovaram redução da mortalidade por câncer de mama com o ensino do autoexame como método formal de rastreamento.

Além disso, foram observados possíveis danos, como aumento de intervenções desnecessárias, exames adicionais e biópsias com resultados benignos.

Assim, o autoexame formal, padronizado e periódico, não deve ser apresentado como estratégia efetiva de rastreamento.

### 5.4 Autoexame não é o mesmo que conscientização corporal

A diretriz diferencia o autoexame formal da estratégia de conscientização corporal.

O fato de o autoexame formal não ser recomendado como rastreamento não significa que a pessoa deva ignorar alterações nas mamas.

A orientação adequada é que a pessoa conheça o próprio corpo, perceba mudanças habituais e procure avaliação profissional diante de alterações suspeitas.

### 5.5 Uso seguro no FemCare AI

O FemCare AI não deve ensinar o autoexame como método obrigatório, periódico ou substituto da mamografia e da avaliação profissional.

O sistema pode explicar:

- que o autoexame formal não é recomendado como método de rastreamento;
- que conhecer o próprio corpo é importante;
- que alterações novas, persistentes ou progressivas devem ser avaliadas;
- que ausência de alteração percebida não exclui doença;
- que sintomas suspeitos devem levar à avaliação profissional.

### 5.6 Formulação segura

Formulação recomendada:

> As diretrizes brasileiras não recomendam o autoexame das mamas como método formal de rastreamento. Ainda assim, é importante conhecer o próprio corpo e procurar avaliação profissional se surgirem alterações novas, persistentes ou suspeitas.

### 5.7 Formulações a evitar

Evitar frases como:

> Faça autoexame todo mês para prevenir câncer.

> O autoexame substitui a mamografia.

> Se você não sentiu nenhum caroço, não precisa se preocupar.

> Autoexame normal descarta câncer.

Essas frases não representam de forma adequada a diretriz e podem gerar falsa segurança.

---

## 6. Estratégia de conscientização

### 6.1 Conceito de conscientização

A diretriz apresenta a estratégia de conscientização, também conhecida como *breast awareness*, como uma abordagem voltada ao conhecimento do próprio corpo e ao reconhecimento de alterações suspeitas nas mamas.

Essa estratégia não deve ser confundida com o autoexame formal das mamas.

Enquanto o autoexame formal envolve técnica padronizada e periodicidade fixa, a conscientização corporal envolve perceber o que é habitual para cada pessoa e procurar atendimento diante de alterações novas, persistentes ou suspeitas.

### 6.2 Objetivo da conscientização

O objetivo da estratégia de conscientização é favorecer o diagnóstico precoce.

Isso envolve:

- conhecer o aspecto habitual das próprias mamas;
- perceber alterações novas ou incomuns;
- valorizar sinais e sintomas suspeitos;
- procurar serviço de saúde quando houver alteração;
- reduzir atraso entre percepção do sintoma e avaliação profissional.

A diretriz recomenda a implementação de estratégias de conscientização para o diagnóstico precoce do câncer de mama.

### 6.3 Diferença entre conscientização e rastreamento

A conscientização não é um exame de rastreamento.

Ela não substitui mamografia quando o rastreamento está indicado, nem substitui avaliação profissional quando há sintomas.

A conscientização tem papel educativo e busca facilitar a procura oportuna por atendimento quando a pessoa percebe alterações suspeitas.

### 6.4 Alterações que devem chamar atenção

A estratégia de conscientização deve orientar a pessoa a ficar atenta a alterações como:

- nódulo ou caroço na mama;
- nódulo ou caroço na axila;
- alteração no formato da mama;
- retração ou afundamento da pele;
- retração ou mudança no formato do mamilo;
- secreção espontânea pelo mamilo;
- secreção sanguinolenta;
- pele com aspecto de casca de laranja;
- aumento progressivo da mama;
- lesões persistentes na pele, aréola ou mamilo.

Essas alterações não significam necessariamente câncer, mas devem motivar avaliação profissional.

### 6.5 Papel dos serviços de saúde

A diretriz destaca que a conscientização só é efetiva se os serviços de saúde estiverem preparados para acolher e avaliar pessoas com sinais ou sintomas suspeitos.

Isso envolve:

- acesso à atenção primária;
- profissionais capacitados;
- avaliação clínica adequada;
- encaminhamento quando necessário;
- confirmação diagnóstica oportuna;
- continuidade do cuidado.

### 6.6 Uso seguro no FemCare AI

No FemCare AI, a estratégia de conscientização pode ser apresentada como orientação educativa.

O sistema pode explicar que a pessoa deve conhecer o próprio corpo, observar alterações e buscar avaliação profissional se notar sinais suspeitos.

O sistema não deve transformar a conscientização em diagnóstico caseiro.

Formulação segura:

> Conhecer o próprio corpo é importante para perceber alterações novas ou persistentes. Se surgir nódulo, secreção sanguinolenta, retração da pele ou do mamilo, caroço na axila ou mudança progressiva na mama, procure avaliação profissional.

---

## 7. Sinais e sintomas suspeitos

### 7.1 Objetivo desta seção

Esta seção resume os sinais e sintomas que a diretriz considera relevantes para encaminhamento e investigação diagnóstica.

O objetivo não é confirmar câncer, mas identificar situações que merecem avaliação profissional.

No contexto do FemCare AI, esses sinais devem ser usados para orientar busca por atendimento, não para gerar diagnóstico.

### 7.2 Sinais e sintomas considerados para referência urgente

A diretriz recomenda que os seguintes sinais e sintomas sejam considerados para referência urgente a serviços de diagnóstico mamário:

- qualquer nódulo mamário em mulheres com mais de 50 anos;
- nódulo mamário em mulheres com mais de 30 anos que persiste por mais de um ciclo menstrual;
- nódulo mamário de consistência endurecida e fixo, ou que vem aumentando de tamanho, em mulheres adultas de qualquer idade;
- descarga papilar sanguinolenta unilateral;
- lesão eczematosa da pele que não responde a tratamentos tópicos;
- homens com mais de 50 anos com tumoração palpável unilateral;
- presença de linfadenopatia axilar;
- aumento progressivo do tamanho da mama com sinais de edema, como pele com aspecto de casca de laranja;
- retração na pele da mama;
- mudança no formato do mamilo.

### 7.3 Nódulo mamário

O nódulo mamário é um dos sinais mais importantes na avaliação de suspeita de câncer de mama.

A diretriz destaca maior atenção para:

- qualquer nódulo em mulheres com mais de 50 anos;
- nódulo em mulheres com mais de 30 anos quando persiste por mais de um ciclo menstrual;
- nódulo endurecido;
- nódulo fixo;
- nódulo que aumenta de tamanho.

A presença de nódulo não confirma câncer, mas deve ser avaliada por profissional de saúde.

### 7.4 Descarga papilar

A descarga papilar sanguinolenta unilateral é considerada sinal que merece referência urgente para investigação diagnóstica.

No uso do FemCare AI, esse achado deve ser tratado como sinal de alerta.

Formulação segura:

> Secreção sanguinolenta unilateral pelo mamilo merece avaliação profissional. Isso não confirma câncer, mas precisa ser investigado.

### 7.5 Alterações da pele

A diretriz inclui entre os sinais suspeitos:

- retração na pele da mama;
- aumento progressivo da mama com edema;
- pele com aspecto de casca de laranja;
- lesão eczematosa que não responde a tratamentos tópicos.

Essas alterações devem ser avaliadas por profissional de saúde, especialmente quando forem novas, persistentes ou progressivas.

### 7.6 Alterações do mamilo

Mudança no formato do mamilo é listada pela diretriz como sinal que merece avaliação.

Alterações relevantes incluem:

- retração recente;
- mudança persistente de formato;
- alteração unilateral;
- secreção sanguinolenta associada;
- lesão persistente em mamilo ou aréola.

O sistema não deve afirmar que uma alteração no mamilo é benigna ou maligna sem avaliação profissional.

### 7.7 Linfadenopatia axilar

A presença de linfadenopatia axilar também é considerada sinal de alerta.

No contexto de orientação, isso pode ser descrito como caroço ou aumento de linfonodo na axila, especialmente quando associado a alterações mamárias.

Formulação segura:

> Caroço persistente na axila, especialmente quando associado a alteração na mama, deve ser avaliado por profissional de saúde.

### 7.8 Sinais em homens

Embora o câncer de mama seja mais frequente em mulheres, a diretriz também menciona homens com mais de 50 anos com tumoração palpável unilateral como situação que merece referência para investigação.

Portanto, alterações mamárias em homens não devem ser ignoradas.

### 7.9 Uso seguro no FemCare AI

Quando o usuário relatar qualquer sinal suspeito, o FemCare AI deve:

- evitar diagnóstico definitivo;
- evitar descartar câncer;
- recomendar avaliação profissional;
- explicar que o sinal merece investigação;
- usar linguagem cuidadosa;
- não prescrever medicamentos;
- não sugerir aguardar indefinidamente.

Formulação segura:

> O sinal descrito merece avaliação profissional. Isso não significa necessariamente câncer, mas deve ser investigado de forma adequada por um serviço de saúde.

---

## 8. Confirmação diagnóstica e referência

### 8.1 Papel da confirmação diagnóstica

A diretriz destaca que, após a identificação de sinais e sintomas suspeitos, é necessário garantir investigação diagnóstica adequada e oportuna.

A confirmação diagnóstica não é feita apenas pela percepção de sintomas, nem por uma conversa, nem por um exame isolado fora de contexto.

Ela depende de avaliação profissional e, quando indicado, de exames complementares.

### 8.2 Atenção primária

A atenção primária tem papel importante na identificação inicial de sinais e sintomas suspeitos.

Segundo a diretriz, profissionais da atenção primária devem estar preparados para:

- acolher pessoas com queixas mamárias;
- realizar anamnese;
- realizar exame clínico das mamas quando aplicável;
- identificar sinais suspeitos;
- classificar necessidade de encaminhamento;
- orientar continuidade do cuidado.

### 8.3 Encaminhamento para serviço de diagnóstico mamário

Quando há sinais ou sintomas suspeitos, a diretriz recomenda referência para serviços de diagnóstico mamário.

A finalidade do encaminhamento é permitir investigação adequada, com acesso a profissionais e exames necessários para confirmação ou exclusão diagnóstica.

No FemCare AI, isso deve ser apresentado em linguagem simples:

> Procure uma unidade de saúde, ginecologista, mastologista ou serviço de diagnóstico mamário para avaliação.

### 8.4 Confirmação diagnóstica em um mesmo centro de referência

A diretriz recomenda que a avaliação diagnóstica do câncer de mama, após identificação de sinais e sintomas suspeitos na atenção primária, seja feita em um mesmo centro de referência quando possível.

Essa organização busca reduzir atrasos, melhorar a continuidade da investigação e evitar que a pessoa precise retornar repetidamente entre serviços diferentes para cada etapa da avaliação.

### 8.5 Investigação diagnóstica não é rastreamento de rotina

Quando uma pessoa apresenta sinais ou sintomas suspeitos, ela não está apenas em contexto de rastreamento.

Nesses casos, a necessidade principal é avaliação diagnóstica.

Portanto, a resposta não deve se limitar a dizer qual é a periodicidade da mamografia.

Formulação segura:

> Como há relato de alteração na mama, a situação precisa de avaliação profissional. A orientação de mamografia de rotina se aplica a pessoas assintomáticas; sintomas suspeitos devem ser investigados.

### 8.6 Tempo e acesso ao cuidado

A diretriz destaca a importância de evitar atrasos na investigação de casos suspeitos.

O acesso oportuno à avaliação profissional e à confirmação diagnóstica é parte essencial das estratégias de diagnóstico precoce.

No FemCare AI, isso deve ser traduzido como recomendação de não ignorar sintomas persistentes ou progressivos.

### 8.7 Uso seguro no FemCare AI

O FemCare AI pode orientar busca por avaliação profissional, mas não pode:

- confirmar diagnóstico;
- excluir câncer;
- interpretar exames de forma definitiva;
- definir tratamento;
- substituir serviço de saúde;
- prometer acesso a serviço;
- regular vaga ou agendamento real.

Formulação segura:

> A confirmação diagnóstica depende de avaliação profissional e exames adequados. O sistema pode orientar sobre sinais de alerta, mas não substitui consulta, exame clínico ou laudo médico.

---

## 9. Limitações da curadoria

### 9.1 Limitação deste arquivo

Este arquivo é uma curadoria resumida e estruturada da diretriz oficial do INCA/Ministério da Saúde.

Ele não contém todo o conteúdo da publicação original.

Foram priorizadas seções úteis para uso em um MVP acadêmico de IA, especialmente:

- conceitos de detecção precoce;
- rastreamento;
- diagnóstico precoce;
- mamografia;
- autoexame;
- conscientização;
- sinais e sintomas suspeitos;
- encaminhamento para avaliação profissional.

### 9.2 Conteúdo não incluído integralmente

Este arquivo não reproduz integralmente:

- metodologia completa da revisão sistemática;
- todos os quadros GRADE;
- todos os apêndices;
- listas completas de artigos avaliados;
- referências bibliográficas completas da diretriz;
- discussão detalhada sobre cada estudo;
- análise extensa de danos do rastreamento;
- avaliação detalhada de termografia, tomossíntese, ressonância ou ultrassonografia;
- discussões de política pública e organização sistêmica em profundidade.

Esses conteúdos permanecem disponíveis na diretriz oficial completa.

### 9.3 Limitação temporal

A fonte principal é uma diretriz publicada em 2015.

Diretrizes médicas podem ser atualizadas ao longo do tempo.

Para uso acadêmico no MVP, a fonte é adequada como referência oficial, mas qualquer uso clínico ou produtivo exigiria revisão por especialistas e comparação com diretrizes mais recentes.

### 9.4 Limitação clínica

Este arquivo não deve ser usado para tomada de decisão clínica individual.

Ele não substitui:

- consulta médica;
- exame físico;
- laudo radiológico;
- biópsia;
- confirmação histopatológica;
- avaliação especializada;
- protocolos institucionais reais.

### 9.5 Limitação para alto risco

A diretriz curada aqui apresentada é voltada principalmente à população de risco padrão.

Pessoas com possível alto risco devem receber avaliação individualizada por profissional de saúde.

Exemplos de situações que podem exigir avaliação individualizada:

- forte histórico familiar de câncer de mama;
- câncer de ovário na família;
- múltiplos familiares acometidos;
- câncer de mama em idade jovem na família;
- mutação genética conhecida;
- histórico pessoal de câncer de mama;
- radioterapia torácica prévia em idade jovem.

### 9.6 Limitação de uso no RAG

Ao ser usado em RAG, este arquivo deve ser recuperado como fonte de conhecimento, não como regra automática de decisão clínica.

O sistema deve combinar esta fonte com:

- instruções de segurança;
- validação de resposta;
- limitação explícita;
- orientação para avaliação profissional;
- logs e auditoria.

### 9.7 Nota de curadoria

Este arquivo resume e reorganiza conteúdo da diretriz oficial para uso em RAG acadêmico.

Ele busca preservar o sentido da fonte original, mas não é uma transcrição integral.

Regras operacionais específicas do FemCare AI devem ficar documentadas separadamente.

---

## 10. Referência bibliográfica

### 10.1 Referência principal

INSTITUTO NACIONAL DE CÂNCER JOSÉ ALENCAR GOMES DA SILVA.  
**Diretrizes para a Detecção Precoce do Câncer de Mama no Brasil.**  
Rio de Janeiro: INCA, 2015.

### 10.2 Referência curta

Diretrizes para a Detecção Precoce do Câncer de Mama no Brasil — INCA/Ministério da Saúde, 2015.

### 10.3 Identificador sugerido para logs e rastreabilidade

`INCA_MS_Diretrizes_Deteccao_Precoce_Cancer_Mama_2015`

### 10.4 Arquivo original recomendado no repositório

O PDF original pode ser mantido em pasta de referência, por exemplo:

`data/reference/diretrizes_deteccao_precoce_cancer_mama_brasil.pdf`

