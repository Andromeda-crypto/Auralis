import streamlit as st

from sdk import run_controller, run_multinode

st.set_page_config(page_title="Auralis Dashboard", layout="wide")
st.title("Auralis Controllers")

tab1, tab2 = st.tabs(["Single Node", "Multi Node"])

with tab1:
    st.subheader("Single Node Run")
    controller = st.selectbox("Controller", ["heuristic", "rule_based", "mpc_lite"], index=1)
    steps = st.slider("Steps", 20, 240, 60, 10)
    seed = st.number_input("Seed", value=123)
    do_plots = st.checkbox("Show Plots", value=True)
    if st.button("Run Single Node"):
        res = run_controller(
            controller_name=controller, steps=int(steps), seed=int(seed), do_plots=do_plots
        )
        st.json({"kpis": res["kpis"], "econ": res["econ"]})

with tab2:
    st.subheader("Multi Node Run")
    controller = st.selectbox("Controller (multi)", ["heuristic", "rule_based", "mpc_lite"], index=1, key="multi_ctrl")
    num_nodes = st.slider("Num Nodes", 2, 12, 3, 1)
    feeder_limit = st.slider("Feeder Import Limit", 0.2, 2.0, 0.8, 0.05)
    steps = st.slider("Steps (multi)", 20, 240, 60, 10, key="multi_steps")
    seed = st.number_input("Seed (multi)", value=2024, key="multi_seed")
    do_plots = st.checkbox("Show Plots (multi)", value=True, key="multi_plots")
    if st.button("Run Multi Node"):
        res = run_multinode(
            num_nodes=int(num_nodes),
            feeder_limit=float(feeder_limit),
            steps=int(steps),
            seed=int(seed),
            controller_name=controller,
            do_plots=do_plots,
        )
        st.json({
            "site_baseline_import": res["site_baseline_import"][:10],
            "site_control_import": res["site_control_import"][:10],
        })


