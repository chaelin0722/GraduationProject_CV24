//simple test 부분

package com.example.server;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.os.Bundle;
import android.os.Handler;
import android.util.DisplayMetrics;
import android.view.Gravity;
import android.view.WindowManager;
import android.widget.TextView;

public class PopUp extends AppCompatActivity {

    TextView txtText;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_pop_up);

        //ui
        txtText = (TextView)findViewById(R.id.txtText);

        Intent intent = getIntent();
        String data = intent.getStringExtra("message");
        txtText.setText(data);  //이부분을 json parsing




    }
}