
import utils.data_converter as dc
import sim.simulate

DATASET_PATH = "data/dataset_custom.pkl"



if __name__ == "__main__":
    # sim.simulate(
    #     steps=50,
    #     observe_steps=5,
    #     interval_ms=100,
    #     dt=33,
    #     field_size=(100,100),
    #     window_size=(28,28),
    #     obj_radius=3,
    #     obj_direction=(1,-1),
    #     noise=1,
    #     show_hist=True
    # )

    # from genetic.main_ga import genetic_search_dataset

    # best, score = genetic_search_dataset(pop_size=6, generations=4,
    #                                     max_samples=300,
    #                                     n_epochs=1,
    #                                     penalty_factor=0.5,
    #                                     target_spikes=1200,
    #                                     dataset_path=DATASET_PATH)  
    # print(best, score)



    # import utils.generate_data

    # print(utils.generate_data.generate_dataset(frames_count=9))

    dc.dataset_dict_to_events(DATASET_PATH, num_ex=5)


