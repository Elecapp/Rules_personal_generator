<template>
    <b-container>
      <b-row>
        <b-col>
          <b-jumbotron>
            <template #header>Explanatory Rules for temporal data using prior knowledge</template>

            <template #lead>
              This web application demonstrates the integration of Explainable Artificial
              Intelligence (XAI) and Visual Analytics (VA) in the context of health crisis
              management, specifically leveraging the T2.2 use case. By combining XAI and VA,
              the app provides transparent, interpretable insights from complex machine
              learning models, empowering healthcare professionals to better understand
              predictions and make informed decisions during critical situations.
              The platform visualizes key factors driving predictions and facilitates
              an interactive exploration of the data, enhancing trust and collaboration
              in addressing health emergencies.
              <hr/>
            </template>

          </b-jumbotron>
        </b-col>
      </b-row>
      <b-row>
        <b-col>
          <h2>Precition of episode COVID-19 Level</h2>
          <p>The model we use in our case study predicts the class of the next
            OVID-19 event (i.e., the level of morbidity) based on preceding disease
            events and levels of population Mobility.</p>
          <p>Insert the feature values for the episode to classify</p>
        </b-col>
      </b-row>
      <b-form @submit="onSubmit">
        <b-row>
          <b-col>
            <h4>Temporal attributes</h4>
          </b-col>
        </b-row>
        <b-row class="mb-4">
          <b-col>
            <b-form-group id="ig-duration"
                          :label="`Duration (days). Current value: ${form.duration}`"
            label-for="i-duration" :description="`Length in days of the event.`">
              <b-form-input id="i-duration" v-model="form.duration" :min="7" :max="147"
              type="range" required></b-form-input>
            </b-form-group>
          </b-col>
          <b-col>
            <b-form-group id="ig-day_passed"
                          :label="`Days passed. Current value: ${form.days_passed}`"
            label-for="i-day_passed" :description="`Days passed since the first date.`">
              <b-form-input id="i-day_passed" v-model="form.days_passed" :min="42" :max="441"
                            type="range" required></b-form-input>
            </b-form-group>
          </b-col>
        </b-row>

        <b-row>
          <b-col sm="12" lg="6">
            <b-row>
          <b-col>
            <h4>COVID-19 Levels</h4>
            <p>Insert the class of COVID-19 incidence in the past weeks</p>
          </b-col>
        </b-row>
            <b-row>

        <b-col>
          <b-form-group id="ig-c_w5" label="COVID Week -5" label-for="i-c_w5">
            <b-form-radio-group id="i-c_w5" v-model="form.c_w5"
                                :options="covidOptions" button-variant="outline-primary"
                                 buttons
            ></b-form-radio-group>
          </b-form-group>
        </b-col>
      </b-row>
      <b-row>
        <b-col>
          <b-form-group id="ig-c_w4" label="COVID Week -4" label-for="i-c_w4">
            <b-form-radio-group id="i-c_w4" v-model="form.c_w4"
                                :options="covidOptions" button-variant="outline-primary"
                                 buttons
            ></b-form-radio-group>
          </b-form-group>
        </b-col>
      </b-row>
      <b-row>
        <b-col>
          <b-form-group id="ig-c_w3" label="COVID Week -3" label-for="i-c_w3">
            <b-form-radio-group id="i-c_w3" v-model="form.c_w3"
                                :options="covidOptions" button-variant="outline-primary"
                                 buttons
            ></b-form-radio-group>
          </b-form-group>
        </b-col>
      </b-row>

          </b-col>
          <b-col sm="12" lg="6">
            <b-row>
          <b-col>
            <h4>Mobility Levels</h4>
            <p>Insert the class of Mobility incidence in the past weeks</p>
          </b-col>
        </b-row>
      <b-row>
        <b-col>
          <b-form-group id="ig-m_w5" label="Mobility Week -5" label-for="i-m_w5">
            <b-form-radio-group id="i-m_w5" v-model="form.m_w5"
                                :options="mobilityOptions" button-variant="outline-primary"
                                 buttons
            ></b-form-radio-group>
          </b-form-group>
        </b-col>
      </b-row>
      <b-row>
        <b-col>
          <b-form-group id="ig-m_w4" label="Mobility Week -4" label-for="i-m_w4">
            <b-form-radio-group id="i-m_w4" v-model="form.m_w4"
                                :options="mobilityOptions" button-variant="outline-primary"
                                 buttons
            ></b-form-radio-group>
          </b-form-group>
        </b-col>
      </b-row>
      <b-row>
        <b-col>
          <b-form-group id="ig-m_w3" label="Mobility Week -3" label-for="i-m_w3">
            <b-form-radio-group id="i-m_w3" v-model="form.m_w3"
                                :options="mobilityOptions" button-variant="outline-primary"
                                 buttons
            ></b-form-radio-group>
          </b-form-group>
        </b-col>
      </b-row>
      <b-row>
        <b-col>
          <b-form-group id="ig-m_w2" label="Mobility Week -2" label-for="i-m_w2">
            <b-form-radio-group id="i-m_w2" v-model="form.m_w2"
                                :options="mobilityOptions" button-variant="outline-primary"
                                buttons
            ></b-form-radio-group>
          </b-form-group>
        </b-col>
      </b-row>
          </b-col>
        </b-row>


      <b-row class="mb-5 mt-5">
        <b-col>
          <b-button type="submit" variant="primary" :disabled="invalid">Submit</b-button>
        </b-col>
      </b-row>
      </b-form>

      <b-row v-if="explanation.length>0">
        <b-col md="8" class="position-relative">
          <b-card no-body class="mb-5">
            <b-card-body v-if="selectedExplanation.explanation.rule.premises">
              <h4>Why the class label is <b-badge variant="primary" class="predicate">
                {{selectedPrediction.prediction}}</b-badge>?</h4>
              <hr/>
              <p>Because all the following conditions happen</p>
              <div class="mb-1">
                <div v-for="(r, idx) in selectedExplanation.explanation.rule.premises"
                  v-bind:key="idx" >
                  <b-badge variant="success"
                           class="ml-1 predicate">
                    <span>{{textualPredicate(r)}}</span>
                  </b-badge>
                  <span v-if="idx < selectedExplanation.explanation.rule.premises.length -1">
                    and </span>
                </div>
              </div>
              <hr/>
            </b-card-body>
            <b-card-body v-if="selectedExplanation.explanation.counterfactuals.length > 0">
              <h4>The risk would have been different
                if one of the following alternatives would hold: </h4>
              <div v-for="(crule, index) in selectedExplanation.explanation.counterfactuals"
              class="mb-3" v-bind:key="index">
                <h5>Alternative {{index+1}}:
                  <b-badge class="ml-1 predicate" variant="primary">
                    {{textualPredicate(crule.consequence)}}
                  </b-badge></h5>
                <span v-for="(cr, idx2) in crule.premises" v-bind:key="idx2">
                  <span v-if="idx2 > 0"> and </span>
                  <b-badge variant="warning" class="ml-1 predicate">
                    <span>{{textualPredicate(cr)}}</span>
                  </b-badge>
                </span>
              </div>
            </b-card-body>
          </b-card>
        </b-col>
        <b-col md="4">
          <b-card no-body class="mb-5">
            <b-card-body class="mb-2">
            </b-card-body>
          </b-card>


        </b-col>
      </b-row>
    </b-container>
</template>

<script>
import { getSingleEndpoint } from '@/axiosInstance';
import PatientView from './PatientView';

export default {
  name: 'PatientClassifier',
  components: {
    PatientView,
  },
  data() {
    return {
      form: {
        c_w5: 'c1',
        c_w4: 'c1',
        c_w3: 'c1',
        m_w5: 'm1',
        m_w4: 'm1',
        m_w3: 'm1',
        m_w2: 'm1',
        duration: 10,
        days_passed: 10,
      },
      covidOptions: [
        { text: 'C1 - Low', value: 'c1' },
        { text: 'C2 - Medium', value: 'c2' },
        { text: 'C3 - High', value: 'c3' },
        { text: 'C4 - Very High', value: 'c4' },
      ],
      mobilityOptions: [
        { text: 'M1 - Low', value: 'm1' },
        { text: 'M2 - Medium', value: 'm2' },
        { text: 'M3 - High', value: 'm3' },
        { text: 'M4 - Very High', value: 'm4' },
      ],
      txtExplanation: 'r = { Creat > 3.94, Age > 76.50, SBP <= 138.00, HR > 74.50 } --> ' +
        '{ Risk: True }\nc = { { Creat <= 1.86 },\n      { Age <= 69.50 } }\n',
      explanation: [],
      prediction: [],
      selected: 0,
      invalid: false,
      responses: {
        q1: 0,
        q2: 0,
        q3: 0,
        n1: '',
        n2: '',
        n3: '',
      },
    };
  },
  methods: {
    onSubmit(event) {
      event.preventDefault();
      // this.invalid = true;
      const endpoints = ['predict', 'explain'];
      const queryParams = {
        week5_covid: this.form.c_w5,
        week4_covid: this.form.c_w4,
        week3_covid: this.form.c_w3,
        week5_mobility: this.form.m_w5,
        week4_mobility: this.form.m_w4,
        week3_mobility: this.form.m_w3,
        week2_mobility: this.form.m_w2,
        duration: this.form.duration,
        days_passed: this.form.days_passed,
      };
      const promises = endpoints.map(e => getSingleEndpoint(queryParams, e));

      Promise.all(promises)
        .then((values) => {
          this.prediction.push(values[0].data);
          this.explanation.push(values[1].data);
          this.selected = this.explanation.length;
          // this.invalid = false;
        },
        );
    },
    range(start, end) {
      const length = end - start;
      return Array.from({ length }, (_, i) => start + i);
    },
    textualPredicate(pred) {
      return `${pred.attr} ${pred.op} ${pred.val}`;
    },
  },
  computed: {
    selectedExplanation() {
      if (this.explanation.length > 0) { return this.explanation[this.selected - 1]; }

      return null;
    },
    selectedPrediction() {
      return this.prediction[this.selected - 1];
    },
  },

};
</script>

<style scoped>
  .predicate{
    font-size: 1.2em;
    padding: 10px 15px;
  }
</style>
