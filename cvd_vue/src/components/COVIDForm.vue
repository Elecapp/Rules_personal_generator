<script>
export default {
  name: 'COVIDForm',
  data() {
    return {
      covid_values: [
        { text: 'Very Low', value: 'c1' },
        { text: 'Low', value: 'c2' },
        { text: 'Medium', value: 'c3' },
        { text: 'High', value: 'c4' },
      ],
      mobility_values: [
        { text: 'Very Low', value: 'm1' },
        { text: 'Low', value: 'm2' },
        { text: 'Medium', value: 'm3' },
        { text: 'High', value: 'm4' },
      ],
      form: {
        week6_covid: 'c1',
        week5_covid: 'c1',
        week4_covid: 'c1',
        week3_covid: 'c1',
        week2_covid: 'c1',
        week6_mobility: 'm1',
        week5_mobility: 'm1',
        week4_mobility: 'm1',
        week3_mobility: 'm1',
        week2_mobility: 'm1',
        week1_mobility: 'm1',
        days_passed: 0,
      },
      neighb_parameters: {
        NeighbTypes: [],
        NumSamples: 3000,
      },
      neighb_types_options: [
        { text: 'Training set', value: 'train' },
        { text: 'Random generator', value: 'random' },
        { text: 'Custom', value: 'custom' },
        { text: 'Genetic', value: 'genetic' },
        { text: 'GPT', value: 'gpt' },
        { text: 'Baseline Train', value: 'baseline' },
      ],
    };
  },
  methods: {
    sendRequest() {
      const request = {
        event: {
          week6_covid: this.form.week6_covid,
          week5_covid: this.form.week5_covid,
          week4_covid: this.form.week4_covid,
          week3_covid: this.form.week3_covid,
          week2_covid: this.form.week2_covid,
          week6_mobility: this.form.week6_mobility,
          week5_mobility: this.form.week5_mobility,
          week4_mobility: this.form.week4_mobility,
          week3_mobility: this.form.week3_mobility,
          week2_mobility: this.form.week2_mobility,
          week1_mobility: this.form.week1_mobility,
          days_passed: this.form.days_passed,
        },
        num_samples: this.neighb_parameters.NumSamples,
        neighborhood_types: this.neighb_types_options
          .filter(v => this.neighb_parameters.NeighbTypes.includes(v.value))
          .map(v => v.value),
      };
      console.log('request', request);
      const strValue = JSON.stringify(request);
      this.$emit('sendRequest', strValue);
    },
  },
};
</script>

<template>
  <b-form>
    <b-row>
      <b-col>
        <b-form-group id="ig-c_w6" label="COVID Level - Week 6" label-for="i-c_w6">
          <b-form-radio-group id="i-c_w6" v-model="form.week6_covid"
                              :options="covid_values" button-variant="outline-primary"
                               buttons
          ></b-form-radio-group>
        </b-form-group>
      </b-col>
      <b-col>
        <b-form-group id="ig-c_w5" label="COVID Level - Week 5" label-for="i-c_w5">
          <b-form-radio-group id="i-c_w5" v-model="form.week5_covid"
                              :options="covid_values" button-variant="outline-primary"
                               buttons
          ></b-form-radio-group>
        </b-form-group>
      </b-col>
      <b-col>
        <b-form-group id="ig-c_w4" label="COVID Level - Week 4" label-for="i-c_w4">
          <b-form-radio-group id="i-c_w4" v-model="form.week4_covid"
                              :options="covid_values" button-variant="outline-primary"
                               buttons
          ></b-form-radio-group>
        </b-form-group>
      </b-col>
    </b-row>
    <b-row>
      <b-col>
        <b-form-group id="ig-c_w3" label="COVID Level - Week 3" label-for="i-c_w3">
          <b-form-radio-group id="i-c_w3" v-model="form.week3_covid"
                              :options="covid_values" button-variant="outline-primary"
                               buttons
          ></b-form-radio-group>
        </b-form-group>
      </b-col>
      <b-col>
        <b-form-group id="ig-c_w2" label="COVID Level - Week 2" label-for="i-c_w2">
          <b-form-radio-group id="i-c_w2" v-model="form.week2_covid"
                              :options="covid_values" button-variant="outline-primary"
                               buttons
          ></b-form-radio-group>
        </b-form-group>
      </b-col>
      <b-col></b-col>
    </b-row>
    <b-row>
      <b-col>
        <b-form-group id="ig-m_w6" label="Mobility Level - Week 6" label-for="i-m_w6">
          <b-form-radio-group id="i-m_w6" v-model="form.week6_mobility"
                              :options="mobility_values" button-variant="outline-warning"
                               buttons
          ></b-form-radio-group>
        </b-form-group>
      </b-col>
      <b-col>
        <b-form-group id="ig-m_w5" label="Mobility Level - Week 5" label-for="i-m_w5">
          <b-form-radio-group id="i-m_w5" v-model="form.week5_mobility"
                              :options="mobility_values" button-variant="outline-warning"
                               buttons
          ></b-form-radio-group>
        </b-form-group>
      </b-col>
      <b-col>
        <b-form-group id="ig-m_w4" label="Mobility Level - Week 4" label-for="i-m_w4">
          <b-form-radio-group id="i-m_w4" v-model="form.week4_mobility"
                              :options="mobility_values" button-variant="outline-warning"
                               buttons
          ></b-form-radio-group>
        </b-form-group>
      </b-col>
    </b-row>
    <b-row>
      <b-col>
        <b-form-group id="ig-m_w3" label="Mobility Level - Week 3" label-for="i-m_w3">
          <b-form-radio-group id="i-m_w3" v-model="form.week3_mobility"
                              :options="mobility_values" button-variant="outline-warning"
                               buttons
          ></b-form-radio-group>
        </b-form-group>
      </b-col>
      <b-col>
        <b-form-group id="ig-m_w2" label="Mobility Level - Week 2" label-for="i-m_w2">
          <b-form-radio-group id="i-m_w2" v-model="form.week2_mobility"
                              :options="mobility_values" button-variant="outline-warning"
                               buttons
          ></b-form-radio-group>
        </b-form-group>
      </b-col>
      <b-col>
        <b-form-group id="ig-m_w1" label="Mobility Level - Week 1" label-for="i-m_w1">
          <b-form-radio-group id="i-m_w1" v-model="form.week1_mobility"
                              :options="mobility_values" button-variant="outline-warning"
                               buttons
          ></b-form-radio-group>
        </b-form-group>
      </b-col>
    </b-row>
    <b-row align-h="center">
      <b-col cols="6">
        <b-form-group id="days-passed-g" label-for="days-passed-i"
                      description="How many days since the beginning of the observations">
          <label for="days-passed-i">Days Passed: {{form.days_passed}}
            ({{Number(form.days_passed / 7).toFixed(2)}} weeks)</label>
          <b-form-input id="days-passed-i" v-model="form.days_passed" required
          type="range" :min="0" :max="7 * 52" :step="1"/>
        </b-form-group>
      </b-col>
    </b-row>
    <hr/>
    <b-row>
      <b-col cols="4">
        <b-form-group id="NumSamples-g" label-for="NumSamples-i"
        description="Number of instances of the neighborhood to generate">
          <label for="NumSamples-i">Neighborhood size: {{ neighb_parameters.NumSamples }}</label>
          <b-form-input id="NumSamples-i" v-model="neighb_parameters.NumSamples" required
                        type="number" :min="0" :max="5000" :step="10"/>
        </b-form-group>
      </b-col>
      <b-col>
        <b-form-group id="NeighbTypes-g" label-for="NeighbTypes-i"
        description="Select one or more neighborhood generation types">
          <label for="NeighbTypes-i">Neighborhood types:</label>
          <b-form-checkbox
            v-for="option in neighb_types_options"
            v-model="neighb_parameters.NeighbTypes"
            :key="option.value"
            :value="option.value"
            name="flavour-4a"
          >
            {{ option.text }}
          </b-form-checkbox>
        </b-form-group>
      </b-col>
    </b-row>
    <b-row class="mb-4">
      <b-col>
        <b-button @click="sendRequest()" variant="primary">Submit</b-button>
      </b-col>
    </b-row>
  </b-form>
</template>

<style scoped>

</style>
