import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from io import BytesIO   # ✅ ADICIONE ESTA LINHA


st.set_page_config(page_title="Insumos do Shopping – Otimização de Estoques", layout="wide")

st.title("Insumos de Shopping – Otimização de Compras com um abordagem clássica EOQ.")

st.header("Mapa da Otimização • Tipos e Ramificações")

with st.expander("1) Por natureza das variáveis"):
    st.markdown(
        """
- **Contínua:** variáveis podem assumir valores reais (ex.: mistura de produtos, Q* do EOQ).
- **Inteira (ILP):** variáveis inteiras (ex.: número de veículos, abrir/fechar CD).
- **Mista (MILP/MINLP):** combinação de contínuas e inteiras (muito comum em logística).
        """
    )

with st.expander("2) Por número de objetivos"):
    st.markdown(
        """
- **Monoobjetivo:** um único critério (minimizar custo total).
- **Multiobjetivo:** vários critérios simultâneos (ex.: custo **e** emissões **e** nível de serviço).
  - Resultado típico: **fronteira de Pareto** (trade-offs eficientes).
        """
    )

with st.expander("3) Por forma da função (e das restrições)"):
    st.markdown(
        """
- **Linear (LP/ILP/MILP):** função e restrições lineares.
- **Não Linear (NLP/MINLP):** presença de termos quadráticos, exponenciais, produtos, etc.
  - **Convexa:** um único ótimo global; resolução mais direta.
  - **Não convexa:** múltiplos ótimos locais; resolução mais difícil.
        """
    )

with st.expander("4) Por natureza do problema"):
    st.markdown(
        """
- **Determinística:** parâmetros conhecidos (sem incerteza explícita).
- **Estocástica:** parâmetros aleatórios (ex.: demanda ~ distribuição).
- **Robusta:** solução boa em vários cenários adversos (piora controlada).
- **Dinâmica:** decisões interdependentes ao longo do tempo (programação dinâmica).
        """
    )

with st.expander("5) Por método de resolução"):
    st.markdown(
        """
- **Analítica (fechada):** solução por **fórmula direta** (ex.: EOQ).
- **Exata:** garante ótimo global (ex.: **Simplex** para LP; **Branch & Bound/Cut** para ILP/MILP).
- **Heurística:** boas soluções rápidas, sem garantia de ótimo (greedy, busca local).
- **Metaheurística:** estratégias gerais para problemas grandes/complexos (Genéticos, Simulated Annealing, Ant Colony, PSO).
- **Híbrida:** combinações (ex.: exato + metaheurística).
        """
    )

with st.expander("6) Por horizonte de tempo"):
    st.markdown(
        """
- **Estática:** decisão pontual (ex.: determinar um lote padrão de compra).
- **Dinâmica (multi-período):** sequência de decisões ao longo do tempo (estoque com revisões periódicas).
        """
    )

st.header("Programação Matemática • Principais Famílias")

col1, col2 = st.columns(2)
with col1:
    st.markdown(
        """
**Programação Linear (LP):**
- Função e restrições lineares
- Variáveis contínuas
- Solver típico: **Simplex/Interior Point**

**Programação Linear Inteira (ILP):**
- Como LP, mas com variáveis inteiras
- Solver: **Branch & Bound/Cut**

**Programação Linear Inteira Mista (MILP):**
- Parte contínua + parte inteira
- Muito usado em **roteirização, localização, planejamento**
        """
    )
with col2:
    st.markdown(
        """
**Programação Não Linear (NLP):**
- Função/restrições não lineares
- Convexa ou não convexa
- Solver: Gradiente/Interior Point/Trust-Region

**MINLP (Não Linear Inteira Mista):**
- NLP + variáveis inteiras
- Muito comum em **processos e supply chain avançado**

**Programação Dinâmica / Estocástica / Robusta:**
- Tratam **tempo** e/ou **incerteza**
        """
    )

st.markdown("Compare o **cenário atual (baseline)** com o **ótimo** e estime a economia.")

# ---------------------------
# ABAS
# ---------------------------
aba = st.radio("Escolha uma seção:", [
    "🔧 Calculadora",
    "📘 Intuição da Modelagem",
    "📂 Etapas da Modelagem Matemática",
    "🧮 Exemplo Numérico",
    "📑 Multi-SKU & Upload"
])

# ---------------------------
# FUNÇÕES AUXILIARES
# ---------------------------
def eoq(D_per, K, h_per):
    """EOQ na base de tempo escolhida (período = semana ou mês)."""
    if D_per <= 0 or K < 0 or h_per <= 0:
        return np.nan
    return np.sqrt((2.0 * K * D_per) / h_per)

def custos_periodicos(Q, D_per, K, h_per, SS=0.0):
    """
    Custo de pedido + custo de posse (com SS) na base do período (semana ou mês).
    Retorna (c_ped, c_pos, c_tot) por período.
    """
    if Q is None or Q <= 0 or D_per <= 0 or h_per < 0 or K < 0:
        return np.nan, np.nan, np.nan
    c_ped = K * D_per / Q
    c_pos = h_per * (Q / 2.0 + max(0.0, SS))
    return c_ped, c_pos, c_ped + c_pos

def lead_time_stats_from_base(D_base, sigma_base, L_dias, base="mensal"):
    """
    Converte demanda média e desvio da base (mensal/semanal) para base diária,
    e calcula mu_L e sigma_L no lead time L (dias). Retorna (mu_L, sigma_L, d_dia).
    """
    if base == "semanal":
        d_dia = D_base / 7.0
        sigma_d_dia = sigma_base / np.sqrt(7.0)
    else:  # mensal
        d_dia = D_base / 30.0
        sigma_d_dia = sigma_base / np.sqrt(30.0)
    mu_L = d_dia * L_dias
    sigma_L = sigma_d_dia * np.sqrt(max(L_dias, 1))
    return mu_L, sigma_L, d_dia, sigma_d_dia

def ajusta_por_moq_multiplo(q, moq=0, mult=0):
    if q is None or np.isnan(q):
        return q
    q_adj = q
    if moq and moq > 0 and q_adj < moq:
        q_adj = moq
    if mult and mult > 0:
        q_adj = mult * np.ceil(q_adj / mult)
    return q_adj

def sim_serrilhado_com_leadtime(Q, r, d_dia, L, sigma_d_dia=0.0, T=180,
                                seed=42, clamp_zero=True, start_mode="steady"):
    """
    Simula política (Q, r) com lead time L e demanda ~ Normal(d_dia, sigma_d_dia).

    start_mode:
      - "steady": começa já em regime, com on_hand=r e 1 pedido emitido que chega em L
      - "lot":    começa com on_hand=Q e sem pedidos (ciclo inicial pode encostar em 0)

    Retorna:
      x (dias), y_onhand, y_pos (posição), order_times, arrival_times, stockout_days
    """
    rng = np.random.default_rng(seed)

    if start_mode == "steady":
        on_hand = float(r)
        pending = [(int(L), Q)]  # pedido emitido agora; chega em L
    else:
        on_hand = float(Q)
        pending = []

    x = [0.0]
    y_onhand = [on_hand]
    y_pos = [on_hand]  # posição inicial = on_hand (sem trânsito)
    order_times = []
    arrival_times = []
    stockout_days = 0

    for t in range(1, T + 1):
        # 1) Chegadas
        if pending:
            arrived = [q for (day, q) in pending if day == t]
            if arrived:
                on_hand += sum(arrived)
                arrival_times.append(t)
                pending = [(day, q) for (day, q) in pending if day != t]

        # 2) Consumo
        demand = d_dia if sigma_d_dia <= 0 else max(0.0, rng.normal(d_dia, sigma_d_dia))
        on_hand -= demand
        if clamp_zero and on_hand < 0:
            on_hand = 0.0
            stockout_days += 1

        # 3) Posição de estoque
        on_order = sum(q for (_, q) in pending) if pending else 0.0
        position = on_hand + on_order

        # 4) Regra (Q, r): emite pedido que chegará em t+L
        if position <= r:
            pending.append((t + int(L), Q))
            order_times.append(t)

        x.append(t)
        y_onhand.append(on_hand)
        y_pos.append(position)

    return x, y_onhand, y_pos, order_times, arrival_times, stockout_days

# ---------------------------
# 🔧 CALCULADORA (1 SKU) – revisada
# ---------------------------
if aba == "🔧 Calculadora":
    base_tempo = st.sidebar.radio(
        "Escolha a base dos seus dados:",
        ["Mensal", "Semanal"], index=0,
        help=("Use a mesma base do seu histórico. "
              "Todos os cálculos (Q*, r, SS e custos) saem nessa base, "
              "e o app também projeta o equivalente anual (12× ou 52×).")
    )
    base_key = "mensal" if base_tempo == "Mensal" else "semanal"
    periodo_label = "mês" if base_key == "mensal" else "semana"
    periods_per_year = 12 if base_key == "mensal" else 52

    st.sidebar.header("Parâmetros – Demanda & Incerteza")
    if base_key == "mensal":
        D_base = st.sidebar.number_input(
            f"Demanda média (un/{periodo_label})",
            value=1500, min_value=1, step=50,
            help="Consumo médio por período (média histórica do item)."
        )
        sigma_base = st.sidebar.number_input(
            f"Desvio da demanda (un/{periodo_label})",
            value=300, min_value=0, step=10,
            help=("Variação da demanda por período (desvio-padrão). "
                  "Quanto maior, maior o estoque de segurança (SS).")
        )
    else:
        D_base = st.sidebar.number_input(
            f"Demanda média (un/{periodo_label})",
            value=600, min_value=1, step=20,
            help="Consumo médio por período (média histórica do item)."
        )
        sigma_base = st.sidebar.number_input(
            f"Desvio da demanda (un/{periodo_label})",
            value=180, min_value=0, step=10,
            help=("Variação da demanda por período (desvio-padrão). "
                  "Quanto maior, maior o estoque de segurança (SS).")
        )

    L_dias = st.sidebar.number_input(
        "Lead time (dias)",
        value=10, min_value=1, step=1,
        help=("Tempo entre emitir o pedido e receber o produto (dias corridos). "
              "Usado para calcular r e SS.")
    )

    st.sidebar.header("Parâmetros – Custos")
    v = st.sidebar.number_input(
        "Preço unitário (R$)",
        value=3.50, min_value=0.0, step=0.10,
        help="Custo de aquisição de 1 unidade (R$ por unidade)."
    )
    i_anual = st.sidebar.number_input(
        "Taxa de carregamento i (ao ano)",
        value=0.25, min_value=0.0, max_value=1.0, step=0.01, format="%.2f",
        help=("Custo percentual para manter estoque parado (capital, espaço, seguro, perdas). "
              "Ex.: 0,25 = 25%/ano. O app converte para custo por "
              f"{periodo_label}: h = (i × v)/" + str(periods_per_year))
    )
    K = st.sidebar.number_input(
        "Custo fixo por pedido K (R$)",
        value=80.0, min_value=0.0, step=5.0,
        help=("Custo administrativo/logístico por pedido (frete, recebimento, conferência etc.). "
              "Quanto maior K, maior tende a ser o lote ótimo Q*.")
    )

    # custo de posse por período (semana/mês)
    h_per = (i_anual * v) / periods_per_year

    st.sidebar.header("Restrições operacionais (opcional)")
    moq = st.sidebar.number_input(
        "MOQ (mínimo do fornecedor)",
        value=0, min_value=0, step=10,
        help=("Quantidade mínima aceita pelo fornecedor. "
              "Se Q* < MOQ, ajustamos para o mínimo.")
    )
    mult = st.sidebar.number_input(
        "Múltiplo de lote (ex.: caixas de 250)",
        value=0, min_value=0, step=10,
        help=("Se o fornecedor vende apenas em múltiplos (caixa/pallet), "
              "arredondamos Q para cima no múltiplo informado.")
    )

    st.sidebar.header("Níveis de serviço")
    SL_opt = st.sidebar.slider(
        "Nível de serviço alvo (Ótimo) – %",
        min_value=80, max_value=99, value=95, step=1,
        help=("Probabilidade de NÃO faltar durante o lead time (define z e SS do cenário ótimo). "
              "Ex.: 95% ⇒ z ≈ 1,645.")
    )
    SL_base = st.sidebar.slider(
        "Nível de serviço Baseline – %",
        min_value=80, max_value=99, value=95, step=1,
        help=("Nível de serviço para o cenário atual (baseline). "
              "Usamos para comparar custos em condições equivalentes de serviço.")
    )

    st.sidebar.header("Baseline (cenário atual)")
    Q_base = st.sidebar.number_input(
        "Lote atual Q (un)",
        value=1800, min_value=1, step=50,
        help="Quantidade normalmente comprada por pedido hoje (prática atual)."
    )
    r_base_input = st.sidebar.number_input(
        "Ponto de pedido atual r (un) – opcional",
        value=0, min_value=0, step=10,
        help=("Se você já usa um gatilho de pedido, informe aqui. "
              "Se deixar 0, o app estima r com base no nível de serviço baseline.")
    )

    st.sidebar.header("Situação atual (opcional)")
    on_hand_now = st.sidebar.number_input(
    "Estoque disponível (on-hand) – agora (un)",
    value=0, min_value=0, step=10,
    help="Unidades fisicamente disponíveis agora."
        )
    
    on_order_now = st.sidebar.number_input(
    "Pedidos em trânsito (on-order) – agora (un)",
    value=0, min_value=0, step=10,
    help="Unidades já pedidas e que ainda não chegaram."
        )
    
    backorders_now = st.sidebar.number_input(
    "Backorders (pedidos pendentes) – agora (un)",
    value=0, min_value=0, step=10,
    help="Unidades já comprometidas e ainda não atendidas.")

    # Posição de estoque (use isto para decidir 'quando pedir')
    posicao_atual = on_hand_now + on_order_now - backorders_now

    # Estatísticas do lead time (a partir da base escolhida)
    mu_L_opt, sigma_L_opt, d_dia, sigma_d_dia = lead_time_stats_from_base(D_base, sigma_base, L_dias, base=base_key)
    z_opt = norm.ppf(SL_opt / 100.0)
    SS_opt = z_opt * sigma_L_opt
    r_opt = mu_L_opt + SS_opt

    z_base = norm.ppf(SL_base / 100.0)
    SS_base_default = z_base * sigma_L_opt
    if r_base_input and r_base_input > 0:
        SS_base = max(0.0, r_base_input - mu_L_opt)
        r_base = r_base_input
    else:
        SS_base = SS_base_default
        r_base = mu_L_opt + SS_base

    # EOQ na base do período e ajustes práticos
    Q_opt_raw = eoq(D_base, K, h_per)
    Q_opt = ajusta_por_moq_multiplo(Q_opt_raw, moq=moq, mult=mult)

    # Custos por período + projeção anual (puramente informativa)
    cped_opt, cpos_opt, ctot_opt = custos_periodicos(Q_opt, D_base, K, h_per, SS=SS_opt)
    cped_base, cpos_base, ctot_base = custos_periodicos(Q_base, D_base, K, h_per, SS=SS_base)
    economia_per = ctot_base - ctot_opt if np.isfinite(ctot_base) and np.isfinite(ctot_opt) else np.nan
    economia_ano = economia_per * periods_per_year if np.isfinite(economia_per) else np.nan
    economia_pct = (economia_per / ctot_base * 100.0) if (np.isfinite(ctot_base) and ctot_base > 0) else np.nan
    
    # ---- Derivados para explicar os custos ----
    n_ped_opt  = D_base / Q_opt if (np.isfinite(Q_opt)  and Q_opt  > 0) else np.nan  # pedidos/periodo (ótimo)
    n_ped_base = D_base / Q_base if (np.isfinite(Q_base) and Q_base > 0) else np.nan  # pedidos/periodo (baseline)

    estoque_medio_opt  = Q_opt/2  + SS_opt  if np.isfinite(Q_opt)  else np.nan
    estoque_medio_base = Q_base/2 + SS_base if np.isfinite(Q_base) else np.nan

    # Componentes que aparecem nas fórmulas
    hold_base_opt  = (Q_opt/2 + SS_opt)   if np.isfinite(Q_opt)  else np.nan
    hold_base_base = (Q_base/2 + SS_base) if np.isfinite(Q_base) else np.nan

    # Cobertura em dias
    cobertura_opt = Q_opt / d_dia if d_dia > 0 and np.isfinite(Q_opt) else np.nan
    cobertura_base = Q_base / d_dia if d_dia > 0 else np.nan

    st.subheader("📈 Resultado – Política (Q, r)")
    colA, colB = st.columns(2)

    with colA:
        st.markdown("### ✅ Ótimo (algoritmo)")
        st.markdown(
        f"- **Quanto pedir (Q\\*)**: {Q_opt:,.0f} un  \n"
        f"  <span style='color:gray'>Tamanho do pedido que minimiza o custo total por {periodo_label}.</span>",
        unsafe_allow_html=True,
    )
        st.markdown(
        f"- **Quando pedir (r)**: {r_opt:,.0f} un  \n"
        f"  <span style='color:gray'>Gatilho: dispare quando a **posição de estoque** ≤ r "
        f"(on-hand + em trânsito − backorders). Inclui demanda média no lead time + SS.</span>",
        unsafe_allow_html=True,
    )
        st.markdown(
        f"- **Estoque de segurança (SS)**: {SS_opt:,.0f} un  \n"
        f"  <span style='color:gray'>‘Almofada’ estatística para cobrir a variabilidade durante o lead time.</span>",
        unsafe_allow_html=True,
    )
        st.markdown(
        f"- **Cobertura do lote**: {cobertura_opt:,.0f} dias  \n"
        f"  <span style='color:gray'>Tempo médio que o lote Q\\* cobre o consumo.</span>",
        unsafe_allow_html=True,
    )
        st.markdown(
        f"- **Custo por {periodo_label} – Pedidos:** R$ {cped_opt:,.0f}  \n"
        f"  <span style='color:gray'>= K·D/Q = {K:,.2f} × {D_base:,.0f} ÷ {Q_opt:,.0f} "
        f"→ {n_ped_opt:,.2f} pedidos/{periodo_label}.</span>",
        unsafe_allow_html=True,
    )
        st.markdown(
        f"- **Custo por {periodo_label} – Posse:** R$ {cpos_opt:,.0f}  \n"
        f"  <span style='color:gray'>= h·(Q/2 + SS) = {h_per:,.4f} × ({Q_opt:,.0f}/2 + {SS_opt:,.0f}) "
        f"= {h_per:,.4f} × {hold_base_opt:,.0f}.</span>",
        unsafe_allow_html=True,
    )
        st.markdown(
        f"- **Custo por {periodo_label} – Total:** **R$ {ctot_opt:,.0f}**  \n"
        f"  <span style='color:gray'>= (Pedidos) + (Posse) = {cped_opt:,.0f} + {cpos_opt:,.0f}.</span>",
        unsafe_allow_html=True,
    )


    with colB:
        st.markdown("### 📌 Baseline (atual)")
        st.markdown(
        f"- **Quanto pedir (Q₀)**: {Q_base:,.0f} un  \n"
        f"  <span style='color:gray'>Tamanho do pedido praticado hoje.</span>",
        unsafe_allow_html=True,
    )
        st.markdown(
        f"- **Quando pedir (r₀)**: {r_base:,.0f} un  \n"
        f"  <span style='color:gray'>Gatilho atual. Se r₀ ≈ μ_L ⇒ SS₀ ≈ 0 (sem margem de segurança).</span>",
        unsafe_allow_html=True,
    )
        st.markdown(
        f"- **Estoque de segurança (SS₀)**: {SS_base:,.0f} un  \n"
        f"  <span style='color:gray'>Reserva do baseline (derivada de r₀ ou do SL baseline).</span>",
        unsafe_allow_html=True,
    )
        st.markdown(f"- **Cobertura do lote**: {cobertura_base:,.0f} dias")
        st.markdown(
        f"- **Custo por {periodo_label} – Pedidos:** R$ {cped_base:,.0f}  \n"
        f"  <span style='color:gray'>= K·D/Q = {K:,.2f} × {D_base:,.0f} ÷ {Q_base:,.0f} "
        f"→ {n_ped_base:,.2f} pedidos/{periodo_label}.</span>",
        unsafe_allow_html=True,
    )
        st.markdown(
        f"- **Custo por {periodo_label} – Posse:** R$ {cpos_base:,.0f}  \n"
        f"  <span style='color:gray'>= h·(Q/2 + SS) = {h_per:,.4f} × ({Q_base:,.0f}/2 + {SS_base:,.0f}) "
        f"= {h_per:,.4f} × {hold_base_base:,.0f}.</span>",
        unsafe_allow_html=True,
    )
        st.markdown(
        f"- **Custo por {periodo_label} – Total:** **R$ {ctot_base:,.0f}**  \n"
        f"  <span style='color:gray'>= (Pedidos) + (Posse) = {cped_base:,.0f} + {cpos_base:,.0f}.</span>",
        unsafe_allow_html=True,
    )
    
    # difs por componente
    delta_ped = (cped_base - cped_opt) if (np.isfinite(cped_base) and np.isfinite(cped_opt)) else np.nan
    delta_pos = (cpos_base - cpos_opt) if (np.isfinite(cpos_base) and np.isfinite(cpos_opt)) else np.nan
    delta_tot = economia_per  # já calculado acima
    pct = economia_pct        # já calculado acima
        

    with st.expander("🔎 Como ler estes resultados"):
        st.markdown(
        "- **Quanto pedir (Q)** = tamanho do lote. No ótimo é **Q\\***; no baseline, **Q₀**.\n"
        "- **Quando pedir (r)** = nível que **dispara** o pedido (**posição** ≤ r).\n"
        "- **SS** = estoque de segurança para o nível de serviço escolhido (ex.: 95% ⇒ z≈1,645).\n"
        "- **Cobertura** = por quantos dias o lote típico cobre o consumo médio.\n"
        "- **Custo – Pedidos** = K·D/Q; **Custo – Posse** = h·(Q/2 + SS).\n"
        "- O **ótimo** equilibra pedido × posse; o **baseline** é a prática atual para comparar."
    )

    st.markdown("---")
    st.subheader("🟢 Ação recomendada (hoje)")

    # Checagens de sanidade
    if not np.isfinite(r_opt):
        st.warning("Não foi possível calcular o ponto de pedido (r). Verifique demanda, variabilidade e lead time.")
    else:
        st.markdown(
            f"**Posição de estoque agora:** {posicao_atual:,.0f} un  "
            f"(on-hand {on_hand_now:,.0f} + em trânsito {on_order_now:,.0f} − backorders {backorders_now:,.0f})"
        )
        # Lote sugerido (usa Q* ajustado; se não disponível, cai para Q_base)
        sug_Q = Q_opt if (Q_opt is not None and np.isfinite(Q_opt)) else Q_base
        if posicao_atual <= r_opt:
            st.success(
                f"**Dispare um novo pedido agora.** Lote sugerido: **Q = {sug_Q:,.0f} un**.  \n"
                f"🧠 Lembrete: o ponto de pedido **r = {r_opt:,.0f}** já inclui a demanda média no lead time + SS."
            )
        else:
            if d_dia and d_dia > 0:
                dias_para_r = max(0.0, (posicao_atual - r_opt) / d_dia)
                st.info(
                    f"**Aguarde** ~{dias_para_r:,.1f} dias para atingir **r = {r_opt:,.0f} un**.  \n"
                    f"Quando cruzar r, emita um pedido de **Q = {sug_Q:,.0f} un** (política ótima)."
                )
            else:
                st.info(
                    f"**Aguarde** até a posição cair para **r = {r_opt:,.0f} un**.  \n"
                    f"Quando cruzar r, emita um pedido de **Q = {sug_Q:,.0f} un** (política ótima)."
                )

    # Explicação rápida
    with st.expander("🔎 Por que usamos a *posição de estoque* para decidir o quando pedir?"):
        st.markdown(
        "- **Posição de estoque** = **disponível (on-hand)** + **em trânsito (on-order)** − **backorders**.  \n"
        "- Na política **(Q, r)**, o gatilho é quando a **posição ≤ r** (não apenas o on-hand).  \n"
        "- Assim você considera o que já está a caminho e o que já está prometido, evitando pedir cedo demais ou tarde demais."
        )
    
        # Escolhe a cor do banner conforme ganho/perda
    if np.isfinite(delta_tot):
        if delta_tot > 0:
            st.success(
            f"**Economia por {periodo_label}: BRL {delta_tot:,.2f} "
            f"({pct:,.2f}%)**  \n"
            f"📅 **Equivalente anual:** BRL {economia_ano:,.0f}"
        )
        elif delta_tot < 0:
            st.warning(
            f"**Custo adicional vs baseline por {periodo_label}: BRL {abs(delta_tot):,.2f} "
            f"({abs(pct):,.2f}%)**  \n"
            f"📅 **Equivalente anual:** BRL {abs(economia_ano):,.0f}"
        )
        else:
            st.info(
            f"**Sem diferença de custo por {periodo_label}.**  \n"
            f"📅 **Equivalente anual:** BRL 0"
        )
    
    # cartões com decomposição
    colE1, colE2, colE3, colE4 = st.columns(4)
    with colE1:
         st.metric(f"Economia por {periodo_label}", f"R$ {delta_tot:,.2f}", delta=f"{pct:,.2f}%")
    with colE2:
        st.metric("Economia em Pedidos", f"R$ {delta_ped:,.2f}")
    with colE3:
        st.metric("Economia em Posse", f"R$ {delta_pos:,.2f}")
    with colE4:
        st.metric("Equivalente anual", f"R$ {economia_ano:,.0f}")

    st.caption(
    "Economia = (Custo Total Baseline − Custo Total Ótimo). "
    "‘Pedidos’: K·D/Q. ‘Posse’: h·(Q/2 + SS)."
        )

    # ---------------- Gráfico de custo vs Q (por período) ----------------
    st.markdown("---")
    st.subheader(f"💰 Custo por {periodo_label.capitalize()} vs. Tamanho do Lote (Q)")

    Q_grid_right = (Q_opt if np.isfinite(Q_opt) else max(Q_base, 100)) * 5
    Q_grid = np.linspace(max(1, 0.2 * (Q_opt if np.isfinite(Q_opt) else Q_base)), Q_grid_right, 200)
    total_grid = []
    for Q in Q_grid:
        _, _, ctot = custos_periodicos(Q, D_base, K, h_per, SS=SS_opt)  # usa SS do alvo para mesmo nível de serviço
        total_grid.append(ctot)

    fig1, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(Q_grid, total_grid, label=f"Custo por {periodo_label} (com SS alvo)")
    if np.isfinite(Q_opt):
        ax1.axvline(Q_opt, linestyle='--', label=f"Q* (Ótimo) = {Q_opt:,.0f}")
    ax1.axvline(Q_base, linestyle=':', label=f"Q₀ (Baseline) = {Q_base:,.0f}")
    ax1.set_xlabel("Tamanho do lote (Q)")
    ax1.set_ylabel(f"Custo por {periodo_label} (R$)")
    ax1.set_title(f"Custo total por {periodo_label} vs. Q")
    ax1.legend()
    st.pyplot(fig1)

    # ---------------- Gráfico serrilhado com lead time ----------------
    st.markdown("---")
    st.subheader("📦 Dinâmica de Estoque (serrilhado) – com lead time")
    T_sim = st.slider("Horizonte de simulação (dias)", min_value=60, max_value=360, value=180, step=30)
    usar_variabilidade = st.checkbox("Incluir variabilidade diária (desvio ≈ σ_base/√dias)", value=True)

    sd_day = sigma_d_dia if usar_variabilidade else 0.0
    
    # ÓTIMO
    x1, y1_on, y1_pos, ord1, arr1, so1 = sim_serrilhado_com_leadtime(
    Q=Q_opt if np.isfinite(Q_opt) else Q_base,
    r=r_opt, d_dia=d_dia, L=L_dias,
    sigma_d_dia=sd_day, T=T_sim, seed=1, start_mode="steady"
    )
    # BASELINE
    x2, y2_on, y2_pos, ord2, arr2, so2 = sim_serrilhado_com_leadtime(
    Q=Q_base, r=r_base, d_dia=d_dia, L=L_dias,
    sigma_d_dia=sd_day, T=T_sim, seed=2, start_mode="steady"
    )

    fig2, ax2 = plt.subplots(figsize=(8, 4))

    # linhas principais
    ax2.plot(x1, y1_on, label="On-hand Ótimo (Q*, r)")
    ax2.plot(x2, y2_on, label="On-hand Baseline (Q₀, r₀)")

    # posição (on-hand + em trânsito − backorders)
    ax2.plot(x1, y1_pos, alpha=0.6, linewidth=1, label="Posição (Ótimo)")
    ax2.plot(x2, y2_pos, alpha=0.6, linewidth=1, label="Posição (Baseline)")

    # marcadores de eventos
    if ord1:
        ax2.scatter(ord1, [r_opt]*len(ord1), marker="v", s=40, label="Pedido emitido (Ótimo)")
    if arr1:
        ax2.scatter(arr1, [r_opt]*len(arr1), marker="*", s=80, label="Chegada (Ótimo)")
    if ord2:
        ax2.scatter(ord2, [r_base]*len(ord2), marker="v", s=40, label="Pedido emitido (Baseline)")
    if arr2:
        ax2.scatter(arr2, [r_base]*len(arr2), marker="*", s=80, label="Chegada (Baseline)")

    # janelas de lead time (sombreamento suave)
    for t0, t1 in zip(ord1, arr1):
        ax2.axvspan(t0, t1, alpha=0.06)
    for t0, t1 in zip(ord2, arr2):
        ax2.axvspan(t0, t1, alpha=0.03)
    
    # referências
    ax2.axhline(r_opt, ls='--', label="r (Ótimo)")
    ax2.axhline(r_base, ls=':',  label="r₀ (Baseline)")
    ax2.axhspan(SS_opt, r_opt, alpha=0.12, label="Zona de SS (Ótimo)")
    ax2.axhline(0, lw=0.8)

    ax2.set_xlabel("Dias")
    ax2.set_ylabel("Unidades (on-hand)")
    ax2.set_title(f"Ciclos de estoque com lead time – base {periodo_label}")
    ax2.legend()
    st.pyplot(fig2)

    # ---------------- KPIs do ciclo ----------------
    def _kpis(y_on, ords, arrs, SS):
        y_on = np.asarray(y_on, dtype=float)
        ords = np.asarray(ords)
        arrs = np.asarray(arrs)
        return {
        "Pedidos no horizonte": int(len(ords)),
        "Intervalo médio entre chegadas (dias)": (float(np.diff(arrs).mean()) if len(arrs) > 1 else np.nan),
        "Estoque médio (un)": float(np.nanmean(y_on)),
        "% dias abaixo de SS": float(100.0 * np.mean(y_on < SS)),
        "% dias com ruptura": float(100.0 * np.mean(y_on <= 0)),
        "Mín on-hand (un)": float(np.nanmin(y_on)),
        "Máx on-hand (un)": float(np.nanmax(y_on)),
        }

    kpi_opt  = _kpis(y1_on, ord1, arr1, SS_opt)
    kpi_base = _kpis(y2_on, ord2, arr2, SS_base)

    colK1, colK2 = st.columns(2)
    with colK1:
        st.markdown("**KPIs do ciclo – Ótimo**")
        st.markdown(
            f"- Pedidos no horizonte: **{kpi_opt['Pedidos no horizonte']}**  \n"
            f"- Intervalo médio entre chegadas: **{kpi_opt['Intervalo médio entre chegadas (dias)']:.1f} dias**  \n"
            f"- Estoque médio: **{kpi_opt['Estoque médio (un)']:.0f} un**  \n"
            f"- % dias abaixo de SS: **{kpi_opt['% dias abaixo de SS']:.1f}%**  \n"
            f"- % dias com ruptura: **{kpi_opt['% dias com ruptura']:.1f}%**  \n"
            f"- Mín on-hand: **{kpi_opt['Mín on-hand (un)']:.0f}** · Máx on-hand: **{kpi_opt['Máx on-hand (un)']:.0f}**"
        )
    with colK2:
        st.markdown("**KPIs do ciclo – Baseline**")
        st.markdown(
            f"- Pedidos no horizonte: **{kpi_base['Pedidos no horizonte']}**  \n"
            f"- Intervalo médio entre chegadas: **{kpi_base['Intervalo médio entre chegadas (dias)']:.1f} dias**  \n"
            f"- Estoque médio: **{kpi_base['Estoque médio (un)']:.0f} un**  \n"
            f"- % dias abaixo de SS: **{kpi_base['% dias abaixo de SS']:.1f}%**  \n"
            f"- % dias com ruptura: **{kpi_base['% dias com ruptura']:.1f}%**  \n"
            f"- Mín on-hand: **{kpi_base['Mín on-hand (un)']:.0f}** · Máx on-hand: **{kpi_base['Máx on-hand (un)']:.0f}**"
    )
# ---------------------------
# 📘 INTUIÇÃO
# ---------------------------
elif aba == "📘 Intuição da Modelagem":
    st.header("📘 Intuição – Estoques de Insumos em Shopping")
    st.markdown("""
- O que estamos modelando?
                
    Se pedimos muito pouco por vez incorremos em custos como frete entre outros e se pedimos muito de uma vez corremos o risco de ficar com estoque parado. Representando assim um trade-off de otimização que justifica a fórmula do Lote Econômico (Economic Order Quantity).

- O papel da incerteza
                
    A demanda flutua logo não é determinística e o lead time não é imediato podendo ocorrer atrasos até o pedido chegar sendo assim temos SS (Security Stock) baseado na variabilidade da demanda e no lead time. Transformamos a variabilidade em parâmetros numéricos como média  e desvio padrão para chegarmos em um nível de serviço desejado que é a probabilidade de não termos ruptura/stockout. Fica claro que diferentes segmentos podem ter política estratégicas diferentes como material de escritório interno que não é crítico se faltar um dia ou supermercado que caso ocorra ruptura posso perder a venda.
                
- Política de controle (Q, r)
                 
    Onde Q define o quanto pedir, ou seja, toda vez que o estoque chega nessa quantidade fixa faz-se um pedido de tamanho Q e o r define quando pedir, ou seja, o nível de estoque que ao ser atingido dispara o pedido de reposição. 
                
- Comparar Ótimo x Baseline
                 
    Em PO não basta achar o ótimo; é essencial comparar com a prática atual (baseline) para mostrar o valor da modelagem.
                
- Generalização (rede de shoppings, múltiplos SKUs)
                
    Passamos do problema de um único item para portfólios inteiros onde temos seu ótimo individual mas a gestão real envolve restrições conjuntas (capacidade de armazenagem, orçamento, transporte).
""")

# ---------------------------
# 📂 ETAPAS FORMAIS
# ---------------------------
elif aba == "📂 Etapas da Modelagem Matemática":
    st.header("📂 Etapas Formais")
    st.markdown("### 1) Objetivo econômico (função objetivo)")
    st.latex(r"\text{Minimizar } C(Q)=\frac{KD}{Q}+\frac{hQ}{2}")
    
    st.markdown("Queremos minimizar o custo total por período (aqui, vamos usar semanas) como soma de dois termos em conflito:" \
    "Onde K é custo fixo por pedido, D é a demanda semanal,Q é o tamanho do lote e h é o custo de manter o produto estocado nesse tempo período."
        )

    st.markdown("### 2) Solução")
    st.latex(r"Q^*=\sqrt{\frac{2KD}{h}}")

    st.markdown("Derivando C(Q) e igualando a zero,obtemos o ponto de mínimo da curva total."
    )


    st.markdown("### 3) Ponto de pedido e SS")
    st.latex(r"\mu_L=\bar d\,L,\quad \sigma_L=\sigma_d\sqrt{L},\quad SS=z\sigma_L,\quad r=\mu_L+SS")

    st.markdown("""Onde temos:     
    μ_L = demanda média durante o lead time.  
    σ_L = incerteza (desvio-padrão) da demanda no lead time.  
    SS = estoque de segurança definido pelo nível de serviço (ex.: 95% → z ≈ 1,65).  
    r = ponto de pedido: nível de estoque em que devemos emitir um novo pedido.  
""")

    st.markdown("### 4) Custo com (Q, r)")
    st.latex(r"C_{\text{semanal}}\approx \frac{K\,D_{\text{sem}}}{Q}+h_{\text{sem}}\left(\frac{Q}{2}+SS\right)")
    st.markdown("""
    (K·D / Q) → custo de pedidos por semana.  
    (h·Q/2) → custo de estocar esse lote médio por semana.  
    (hₛ·SS) → custo de manter o estoque de segurança por semana.  
    O total semanal **equilibra** custos de pedido, posse e segurança.
    """)

    st.markdown("### 5) Comparação Baseline vs Ótimo")
    st.latex(r"\Delta C = C_{\text{baseline}}-C_{\text{ótimo}}")
    st.markdown("""
    C_baseline = custo semanal na política atual de compras.  
    C_ótimo = custo semanal pelo modelo EOQ/(Q,r).  
    ΔC = economia potencial (quanto pode ser reduzido por semana).  
    """)

    # --- Gráfico didático (EXEMPLO SEMANAL) ---
    # Suponha parâmetros semanais para o exemplo:
    # - D: demanda semanal (un/semana)
    # - K: custo por pedido (R$/pedido)
    # - i_ano: taxa de carregamento anual (fração/ano), v: preço unitário (R$)
    #   h_sem = (i_ano * v) / 52  -> custo de posse por unidade por semana
    D_sem = 2500.0
    K = 120.0
    i_ano = 0.25
    v = 4.00
    h_sem = (i_ano * v) / 52.0  # conversão para base semanal

    Q = np.linspace(max(1, 0.05*D_sem), 8*D_sem, 400)  # grade de Q (un)
    C_pedido = K * D_sem / Q                 # R$/semana
    C_posse  = h_sem * Q / 2.0               # R$/semana
    C_total  = C_pedido + C_posse            # R$/semana

    Q_opt = np.sqrt(2 * K * D_sem / h_sem)   # EOQ em unidades (base semanal)
    C_opt = K * D_sem / Q_opt + h_sem * Q_opt / 2.0

    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(Q, C_pedido, '--', label="Custo de Pedido (R$/sem)")
    ax.plot(Q, C_posse, '--', label="Custo de Posse (R$/sem)")
    ax.plot(Q, C_total, '-',  label="Custo Total (R$/sem)", linewidth=2)
    ax.axvline(Q_opt, linestyle=':', label=f"Q* ≈ {Q_opt:,.0f} un")
    ax.scatter([Q_opt], [C_opt], zorder=5)
    ax.set_xlabel("Quantidade por Pedido (Q)  [un]")
    ax.set_ylabel("Custo por Semana  [R$]")
    ax.set_title("Curvas de Custo (base semanal) e Ponto Ótimo (EOQ)")
    ax.legend()
    st.pyplot(fig)

    st.markdown("O ponto ótimo é aquele em que o custo do pedido é igual ao custo de manter o sku estocado e isso garante o custo total mínimo.")


# ---------------------------
# 🧮 EXEMPLO NUMÉRICO
# ---------------------------
elif aba == "🧮 Exemplo Numérico":
    st.header("🧮 Contexto Jair (Gerente do Shopping Aurora)")

    st.markdown("""
**História**  
Jair é gerente de operações do Shopping Aurora. Ele precisa garantir detergente semanalmente para banheiros,
praça de alimentação e áreas técnicas. Hoje, o time compra em lotes grandes **sem um ponto de pedido formal**.
Jair quer saber quanto deveria pedir **por semana** e **quando** disparar o pedido para reduzir custos e evitar faltas.

**Parâmetros (semanais)**  
- Demanda média semanal: **D_sem = 600 un/sem**  
- Desvio semanal: **σ_sem = 180 un/sem**  
- Lead time: **L = 10 dias**  
- Custo por pedido: **K = R$ 200**  
- Preço unitário: **v = R$ 5,00**  
- Taxa de carregamento anual: **i = 30%/ano** ⇒ custo de posse semanal: **h_sem = (i * v)/52 = R$ 0,02885 /un/sem**  
- Nível de serviço (ótimo e baseline): **95%** (z ≈ 1,645)  
- **Baseline atual**: **Q₀ = 5.000 unidades; ponto de pedido não formalizado
""")

    # --- Cálculos do caso (base SEMANAL) ---
    import math
    from scipy.stats import norm

    D_sem = 600.0
    sigma_sem = 180.0
    L_dias = 10.0
    K = 200.0
    v = 5.0
    i_ano = 0.30
    h_sem = (i_ano * v) / 52.0               # R$/un/sem
    z = norm.ppf(0.95)                       # nível de serviço 95%

    # EOQ (quanto pedir) – base semanal
    Q_opt = math.sqrt(2 * K * D_sem / h_sem)

    # Ponto de pedido e SS (quando pedir)
    d_bar = D_sem / 7.0                      # un/dia
    sigma_d = sigma_sem / math.sqrt(7.0)     # un/dia
    mu_L = d_bar * L_dias
    sigma_L = sigma_d * math.sqrt(L_dias)
    SS = z * sigma_L
    r = mu_L + SS

    # Baseline
    Q_base = 5000.0
    # Para comparar em mesmo nível de serviço, usamos o mesmo SS calculado acima
    def custo_semanal(Q, K, D_sem, h_sem, SS):
        C_ped = K * D_sem / Q
        C_pos = h_sem * (Q / 2.0 + SS)
        return C_ped, C_pos, (C_ped + C_pos)

    Cped_opt, Cpos_opt, Ctot_opt = custo_semanal(Q_opt, K, D_sem, h_sem, SS)
    Cped_base, Cpos_base, Ctot_base = custo_semanal(Q_base, K, D_sem, h_sem, SS)

    economia_abs_sem = Ctot_base - Ctot_opt
    economia_pct = (economia_abs_sem / Ctot_base) * 100.0
    economia_abs_ano = economia_abs_sem * 52.0

    # Coberturas (dias / semanas)
    cobertura_dias_opt = Q_opt / d_bar
    cobertura_sem_opt  = cobertura_dias_opt / 7.0
    cobertura_dias_base = Q_base / d_bar
    cobertura_sem_base  = cobertura_dias_base / 7.0

    # --- Saída formatada ---
    st.subheader("🎯 Política Ótima (Q, r)")
    st.markdown(f"- **Lote econômico/Pedido ótimo (Q\\*):** {Q_opt:,.0f} unidades")
    st.markdown(f"- **Estoque de segurança (SS):** {SS:,.0f} unidades posso adicionar ao pedido para evitar ruptura")
    st.markdown(f"- **Ponto de pedido (r):** {r:,.0f} unidades")
    st.markdown(f"- **Cobertura do lote (ótimo):** {cobertura_dias_opt:,.1f} dias (~{cobertura_sem_opt:,.1f} semanas)")

    st.subheader("📦 Baseline (prática atual)")
    st.markdown(f"- **Lote atual (Q₀):** {Q_base:,.0f} unidades")
    st.markdown(f"- **Cobertura do lote (baseline):** {cobertura_dias_base:,.1f} dias (~{cobertura_sem_base:,.1f} semanas)")

    st.subheader("💰 Custo Semanal – Comparação")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Ótimo (mesmo SL)**")
        st.markdown(f"- Custo de pedidos/sem: **R$ {Cped_opt:,.2f}**")
        st.markdown(f"- Custo de posse/sem: **R$ {Cpos_opt:,.2f}**")
        st.markdown(f"- **Total/sem:** **R$ {Ctot_opt:,.2f}**")
    with col2:
        st.markdown("**Baseline**")
        st.markdown(f"- Custo de pedidos/sem: **R$ {Cped_base:,.2f}**")
        st.markdown(f"- Custo de posse/sem: **R$ {Cpos_base:,.2f}**")
        st.markdown(f"- **Total/sem:** **R$ {Ctot_base:,.2f}**")

        st.success(
        f"✅ **Economia semanal estimada: BRL {economia_abs_sem:,.2f} "
        f"(**{economia_pct:,.2f}%**)  \n"
        f"📅 Equivalente a ~ BRL {economia_abs_ano:,.0f}/ano**"
            )

    st.markdown("""
**Leitura do resultado:**  
- O baseline compra **demais por pedido** (Q₀ bem acima do ótimo), o que **aumenta o custo de posse**.
- A política ótima **(Q\\*, r)** reduz o custo total semanal em **≈ 12%**, sem piorar o nível de serviço (mesmo z).
    """)

    # Gráfico opcional (custo x Q) para visualizar o ganho neste caso
    import numpy as np
    Q_grid = np.linspace(max(200, 0.2*Q_opt), 2.5*Q_base, 250)
    C_grid = [custo_semanal(q, K, D_sem, h_sem, SS)[2] for q in Q_grid]

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(Q_grid, C_grid, label="Custo semanal total (com SS)")
    ax.axvline(Q_opt, ls='--', label=f"Q* ≈ {Q_opt:,.0f}")
    ax.axvline(Q_base, ls=':', label=f"Q₀ (baseline) = {Q_base:,.0f}")
    ax.set_xlabel("Tamanho do lote (Q)")
    ax.set_ylabel("Custo por semana (R$)")
    ax.set_title("Custo semanal vs. Q — Ótimo x Baseline")
    ax.legend()
    st.pyplot(fig)

# ---------------------------
# 📑 MULTI-SKU & UPLOAD
# ---------------------------
elif aba == "📑 Multi-SKU & Upload":
    st.header("📑 Multi-SKU – Upload CSV/Excel e Ranking de Economia")

    # -----------------------------
    # 📤 Upload (somente Excel)
    # -----------------------------
    st.markdown("Faça upload do **Excel (.xlsx/.xls)** exportado do ERP com as colunas padrão.")
    with st.expander("📄 Esquema de colunas (ERP) – obrigatórias e opcionais"):
        st.markdown(
            """
**Obrigatórias (usar exatamente esses nomes):**  
- `SKU -> Nome do item (ex.: Detergente, Papel Higiênico)`  
- `Demanda_mensal` **ou** `Demanda_semanal` (conforme a base escolhida acima) -> Consumo médio  
- `Desvio_mensal` **ou** `Desvio_semanal`  -> Variação da demanda (quanto oscila em relação à média)
- `Preco_unitario` -> Custo unitário do item
- `Taxa_carrying_anual` -> Percentual anual de custo de armazenagem
- `Custo_pedido`  -> Custo fixo por cada pedido feito ao fornecedor (logística, emissão de NF etc.)
- `Lead_time_dias` -> Tempo médio de entrega (quantos dias o fornecedor demora)
- `MOQ`  -> Quantidade mínima obrigatória para compra (se não houver, usar 0)
- `Multiplo`  -> Tamanho do lote de compra (ex.: só vende em caixas de 500)
- `SL` (nível de serviço, %) -> Quanto risco de falta você aceita (ex.: 95% = aceito faltar em 5% dos casos)

**Opcionais:**  
- `Q_base` -> Lote de compra atual que a empresa já usa  
- `r_base` -> Ponto de pedido atual (nível de estoque no qual a empresa costuma disparar o pedido)

**KPIs:**
- `Q*` -> Quantidade ideal para pedir de uma só vez equilibrando, custo de pedir muitas vezes com o custo de manter estoque parado.Se pedir pouco demais, gasta-se muito com pedidos; se pedir demais, gasta muito guardando.

- `r` -> É o estoque mínimo seguro que trata do nosso ponto de ressuprimento, quando o estoque chega nesse nível, é hora de pedir de novo.

- `SS` -> É um estoque extra para absorver imprevistos. 
"""

        )
        base_tempo_multi = st.sidebar.radio(
        "Base dos campos de demanda/desvio do ERP",
        ["Mensal", "Semanal"], index=0,
        help="Selecione se as colunas do ERP estão em base mensal ou semanal."
    )
    base_key_multi = "mensal" if base_tempo_multi == "Mensal" else "semanal"
    periods_per_year_multi = 12 if base_key_multi == "mensal" else 52
    periodo_label_multi = "mês" if base_key_multi == "mensal" else "semana"

    aplicar_restricoes = st.sidebar.checkbox("Aplicar MOQ e Múltiplo do ERP", value=True)

    heuristica_baseline = st.sidebar.selectbox(
        "Se NÃO vier Q_base/r_base no Excel, baseline será:",
        [
            f"Q₀ = Demanda por {periodo_label_multi}, r₀ = μ_L + SS",
            f"Q₀ = 2× Demanda por {periodo_label_multi}, r₀ = μ_L + SS",
            "Q₀ = Q* ajustado por restrições (MOQ/Múltiplo), r₀ = μ_L + SS"
        ],
        help="Regra de comparação se seu ERP não fornecer o lote/ponto de pedido atuais."
    )

    up = st.file_uploader("Solte aqui o Excel do ERP (.xlsx, .xls)", type=["xlsx", "xls"], key="upload_multi")
    if not up:
        st.info("🧭 Aguardando o Excel do ERP…")
        st.stop()

    def read_excel_safely(file):
        name = (file.name or "").lower()
        if name.endswith(".xlsx"):
            try:
                import openpyxl  # noqa: F401
                return pd.read_excel(file, engine="openpyxl")
            except ImportError:
                st.warning("⚠️ 'openpyxl' não encontrado. Tentando engine padrão do pandas.")
                return pd.read_excel(file)
            except Exception as e:
                st.warning(f"Falha com 'openpyxl': {e}. Tentando engine padrão.")
                return pd.read_excel(file)
        elif name.endswith(".xls"):
            try:
                import xlrd  # noqa: F401
                return pd.read_excel(file, engine="xlrd")
            except ImportError:
                st.error("❌ Para .xls é necessário 'xlrd' (pip install xlrd).")
                st.stop()
            except Exception as e:
                st.error(f"❌ Não foi possível ler o .xls (xlrd): {e}")
                st.stop()
        else:
            st.error("Formato não suportado. Use .xlsx ou .xls.")
            st.stop()

    try:
        df_raw = read_excel_safely(up)
    except Exception as e:
        st.error(f"❌ Não foi possível ler o Excel: {e}")
        st.stop()

    required_common = [
        "SKU", "Preco_unitario", "Taxa_carrying_anual",
        "Custo_pedido", "Lead_time_dias", "MOQ", "Multiplo", "SL"
    ]
    required_specific = ["Demanda_mensal", "Desvio_mensal"] if base_key_multi == "mensal" else ["Demanda_semanal", "Desvio_semanal"]
    required_cols = required_common + required_specific
    missing = [c for c in required_cols if c not in df_raw.columns]
    if missing:
        st.error(f"Colunas faltando no Excel do ERP: {missing}")
        st.stop()

    has_Qbase = "Q_base" in df_raw.columns
    has_rbase = "r_base" in df_raw.columns

    def eoq(D_per, K, h_per):
        if D_per is None or D_per <= 0 or K < 0 or h_per <= 0: return np.nan
        return np.sqrt((2.0 * K * D_per) / h_per)

    def custos_periodicos(Q, D_per, K, h_per, SS=0.0):
        if Q is None or Q <= 0 or D_per <= 0 or h_per < 0 or K < 0: return np.nan, np.nan, np.nan
        c_ped = K * D_per / Q
        c_pos = h_per * (Q / 2.0 + max(0.0, SS))
        return c_ped, c_pos, c_ped + c_pos

    def lead_time_stats_from_base(D_base, sigma_base, L_dias, base="mensal"):
        if base == "semanal":
            d_dia = D_base / 7.0; sigma_d_dia = sigma_base / np.sqrt(7.0)
        else:
            d_dia = D_base / 30.0; sigma_d_dia = sigma_base / np.sqrt(30.0)
        mu_L = d_dia * L_dias
        sigma_L = sigma_d_dia * np.sqrt(max(L_dias, 1))
        return mu_L, sigma_L, d_dia, sigma_d_dia

    def ajusta_por_moq_multiplo(q, moq=0, mult=0):
        if q is None or np.isnan(q): return q
        q_adj = float(q)
        if moq and moq > 0 and q_adj < moq: q_adj = float(moq)
        if mult and mult > 0: q_adj = float(mult) * np.ceil(q_adj / float(mult))
        return q_adj

    D_col = "Demanda_mensal" if base_key_multi == "mensal" else "Demanda_semanal"
    S_col = "Desvio_mensal"  if base_key_multi == "mensal" else "Desvio_semanal"

    df = pd.DataFrame({
        "SKU": df_raw["SKU"].astype(str),
        "D_per": pd.to_numeric(df_raw[D_col], errors="coerce"),
        "sigma_per": pd.to_numeric(df_raw[S_col], errors="coerce"),
        "v": pd.to_numeric(df_raw["Preco_unitario"], errors="coerce"),
        "i_anual": pd.to_numeric(df_raw["Taxa_carrying_anual"], errors="coerce"),
        "K": pd.to_numeric(df_raw["Custo_pedido"], errors="coerce"),
        "L_dias": pd.to_numeric(df_raw["Lead_time_dias"], errors="coerce"),
        "MOQ": pd.to_numeric(df_raw["MOQ"], errors="coerce"),
        "Multiplo": pd.to_numeric(df_raw["Multiplo"], errors="coerce"),
        "SL": pd.to_numeric(df_raw["SL"], errors="coerce"),
        "Q_base": pd.to_numeric(df_raw["Q_base"], errors="coerce") if has_Qbase else np.nan,
        "r_base": pd.to_numeric(df_raw["r_base"], errors="coerce") if has_rbase else np.nan,
    })
    df[["MOQ", "Multiplo"]] = df[["MOQ", "Multiplo"]].fillna(0.0)

    results = []
    for _, r in df.iterrows():
        SKU = r["SKU"]; D_per = float(r["D_per"]); sigma_per = float(r["sigma_per"])
        v = float(r["v"]); i_anual = float(r["i_anual"]); K = float(r["K"])
        L = float(r["L_dias"]); SL = float(r["SL"]); moq = float(r["MOQ"]); mult = float(r["Multiplo"])
        Qb_in = r["Q_base"]; rb_in = r["r_base"]

        h_per = (i_anual * v) / periods_per_year_multi
        mu_L, sigma_L, d_dia, sigma_d_dia = lead_time_stats_from_base(D_per, sigma_per, L, base=base_key_multi)
        z = norm.ppf(np.clip(SL / 100.0, 0.01, 0.999))
        SS_opt = z * sigma_L; r_opt = mu_L + SS_opt

        Q_opt_raw = eoq(D_per, K, h_per)
        Q_opt = ajusta_por_moq_multiplo(Q_opt_raw, moq if aplicar_restricoes else 0, mult if aplicar_restricoes else 0)

        if pd.notna(Qb_in) and Qb_in > 0:
            Q_base = float(Qb_in)
        else:
            if heuristica_baseline.startswith("Q₀ = Demanda por"): Q_base = D_per
            elif heuristica_baseline.startswith("Q₀ = 2×"):        Q_base = 2.0 * D_per
            else:
                Q_base = ajusta_por_moq_multiplo(Q_opt_raw, moq if aplicar_restricoes else 0, mult if aplicar_restricoes else 0)
        if aplicar_restricoes:
            Q_base = ajusta_por_moq_multiplo(Q_base, moq, mult)

        if pd.notna(rb_in) and rb_in > 0:
            r_base = float(rb_in); SS_base = max(0.0, r_base - mu_L)
        else:
            SS_base = z * sigma_L; r_base = mu_L + SS_base

        cped_opt, cpos_opt, ctot_opt = custos_periodicos(Q_opt, D_per, K, h_per, SS=SS_opt)
        cped_base, cpos_base, ctot_base = custos_periodicos(Q_base, D_per, K, h_per, SS=SS_base)

        economia_per = ctot_base - ctot_opt if np.isfinite(ctot_base) and np.isfinite(ctot_opt) else np.nan
        economia_ano = economia_per * periods_per_year_multi if np.isfinite(economia_per) else np.nan
        economia_pct = (economia_per / ctot_base * 100.0) if (np.isfinite(ctot_base) and ctot_base > 0) else np.nan

        results.append({
            "SKU": SKU,
            f"Demanda_{periodo_label_multi}": D_per,
            f"σ_{periodo_label_multi}": sigma_per,
            "v (R$)": v,
            "i anual": i_anual,
            "K (R$)": K,
            "L (dias)": L,
            "SL (%)": SL,
            "MOQ": moq,
            "Múltiplo": mult,
            "Q* (ótimo)": np.round(Q_opt, 2),
            "r (ótimo)": np.round(r_opt, 2),
            "SS (ótimo)": np.round(SS_opt, 2),
            "Q₀ (baseline)": np.round(Q_base, 2),
            "r₀ (baseline)": np.round(r_base, 2),
            f"Custo_{periodo_label_multi}_ótimo (R$)": np.round(ctot_opt, 2),
            f"Custo_{periodo_label_multi}_baseline (R$)": np.round(ctot_base, 2),
            f"Economia_{periodo_label_multi} (R$)": np.round(economia_per, 2),
            "Economia_anual (R$)": np.round(economia_ano, 2),
            "Economia (%)": np.round(economia_pct, 2),
        })

    df_out = pd.DataFrame(results)

    st.subheader("🏆 Ranking de Economia (maior → menor)")
    ordenar_por = st.selectbox(
        "Ordenar por:",
        [f"Economia_{periodo_label_multi} (R$)", "Economia_anual (R$)", "Economia (%)"],
        index=1, key="ordenar_por_multi"
    )
    df_rank = df_out.sort_values(by=ordenar_por, ascending=False).reset_index(drop=True)

    filtro_texto = st.text_input("Filtrar por SKU (contém):", "", key="filtro_sku_multi")
    if filtro_texto:
        df_rank = df_rank[df_rank["SKU"].str.contains(filtro_texto, case=False, na=False)]

    st.dataframe(df_rank, use_container_width=True)

    colk1, colk2, colk3 = st.columns(3)
    soma_econ_per = pd.to_numeric(df_rank[f"Economia_{periodo_label_multi} (R$)"], errors="coerce").replace([np.inf, -np.inf], np.nan).sum(skipna=True)
    soma_econ_ano = pd.to_numeric(df_rank["Economia_anual (R$)"], errors="coerce").replace([np.inf, -np.inf], np.nan).sum(skipna=True)
    med_econ_pct = pd.to_numeric(df_rank["Economia (%)"], errors="coerce").replace([np.inf, -np.inf], np.nan).mean(skipna=True)

    colk1.metric(f"Economia total por {periodo_label_multi}", f"R$ {soma_econ_per:,.2f}")
    colk2.metric("Economia total anual", f"R$ {soma_econ_ano:,.2f}")
    colk3.metric("Economia média (%)", f"{med_econ_pct:,.2f}%")

    st.caption("📌 Economia = Custo Baseline − Custo Ótimo (mesmo SL). Custo = K·D/Q + h·(Q/2 + SS).")

    buf_xlsx = BytesIO()
    engine_name = "xlsxwriter"
    try:
        import xlsxwriter
    except Exception:
        engine_name = "openpyxl"
        st.warning("Usando engine 'openpyxl' para exportação (XlsxWriter indisponível).")

    try:
        with pd.ExcelWriter(buf_xlsx, engine=engine_name) as writer:
            df_rank.to_excel(writer, index=False, sheet_name="ranking")
            df_out.to_excel(writer, index=False, sheet_name="calculos")
    except Exception as e:
        st.warning(f"Falha ao usar engine '{engine_name}'. Tentando engine padrão. Detalhe: {e}")
        buf_xlsx = BytesIO()
        with pd.ExcelWriter(buf_xlsx) as writer:
            df_rank.to_excel(writer, index=False, sheet_name="ranking")
            df_out.to_excel(writer, index=False, sheet_name="calculos")

    st.download_button(
        label="💾 Baixar resultado (Excel)",
        data=buf_xlsx.getvalue(),
        file_name="resultado_multi_sku.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="download_multi"
    )