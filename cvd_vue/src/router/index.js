import Vue from 'vue';
import Router from 'vue-router';
import PatientClassifier from '@/components/PatientClassifier';

Vue.use(Router);

export default new Router({
  routes: [
    {
      path: '/',
      name: 'PatientClassifier',
      component: PatientClassifier,
    },
  ],
});
